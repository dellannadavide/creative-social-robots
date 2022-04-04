import skfuzzy as fuzz
from skfuzzy import control as ctrl
from skfuzzy.control.controlsystem import CrispValueCalculator

import utils.constants as Constants


class FuzzyController:
    dimensions_values = {}
    variables_universe = {}
    fuzzysets_values = {}
    variables_default_val = {}

    @staticmethod
    def cleanClassVar():
        # del FuzzyController.dimensions_values
        # del FuzzyController.variables_universe
        # del FuzzyController.fuzzysets_values
        # del FuzzyController.variables_default_val
        # gc.collect()
        FuzzyController.dimensions_values = {}
        FuzzyController.variables_universe = {}
        FuzzyController.fuzzysets_values = {}
        FuzzyController.variables_default_val = {}

    @staticmethod
    def initFuzzyControllersStaticInfo(dimensions_values, variables_universe, fuzzysets_values, variables_default_val):
        FuzzyController.cleanClassVar()
        FuzzyController.dimensions_values = dimensions_values
        FuzzyController.variables_universe = variables_universe
        FuzzyController.fuzzysets_values = fuzzysets_values
        FuzzyController.variables_default_val = variables_default_val

    def __init__(self, rulebase):
        # TODO for now it assumes triangula membership functions
        # TODO it is not very clean and clear code, need to so some cleaning
        self.inputs = []

        self.updateController(rulebase)
        self.rules_str = []
        for r in rulebase:
            self.rules_str.append(str(r))

    def refreshInputs(self):
        self.inputs = [i for i in self.controlsystem._get_inputs()]

    def feed_inputs(self, inputs):
        for v in self.inputs:
            try:
                self.controlsystem.input[v] = inputs[v]
            except:
                pass

    def updateController(self, rulebase):
        self.control = ctrl.ControlSystem(rulebase)
        self.controlsystem = ctrl.ControlSystemSimulation(self.control, cache=False)
        self.refreshInputs()

    """ 
    RE-IMPLEMENTATION OF THE compute() method from skfuzzy so that I can catch all the exceptions and still work in case some outputs are not retrieved
    """

    def compute(self, verbose=False):
        rules_activations = {}
        c_out = {}
        is_exception = False

        self.controlsystem.input._update_to_current()
        if verbose>Constants.VERBOSE_BASIC:
            print("assessor inputs")
            print(self.controlsystem.input)
        # Check if any fuzzy variables lack input values and fuzzify inputs
        for antecedent in self.controlsystem.ctrl.antecedents:
            if antecedent.input[self.controlsystem] is None:
                print("All antecedents must have input values!")
                raise ValueError("All antecedents must have input values!")
            CrispValueCalculator(antecedent, self.controlsystem).fuzz(antecedent.input[self.controlsystem])

        # Calculate rules, taking inputs and accumulating outputs
        first = True
        for rule in self.controlsystem.ctrl.rules:
            # Clear results of prior runs from Terms if needed.
            if first:
                for c in rule.consequent:
                    c.term.membership_value[self.controlsystem] = None
                    c.activation[self.controlsystem] = None
                first = False
            self.controlsystem.compute_rule(rule)
            if verbose>Constants.VERBOSE_BASIC:
                print("antecedent membership of rule " + str(rule))
                print(rule.antecedent.membership_value[self.controlsystem])
            rules_activations[str(rule)] = rule.aggregate_firing[self.controlsystem]  # this is also added

        for consequent in self.controlsystem.ctrl.consequents:
            try:  # here's the difference now. I'm introducing this try-catch test
                consequent.output[self.controlsystem] = CrispValueCalculator(consequent, self.controlsystem).defuzz()
                self.controlsystem.output[consequent.label] = consequent.output[self.controlsystem]
                c_out[consequent.label] = self.controlsystem.output[consequent.label]
                # print("defuzz for "+str(consequent.label)+": "+str(c_out[consequent.label]))
            except:
                # in case it fails to defuzzify because the area is 0, then I give the default value to the variable
                # print("Zero defuzz area for "+str(consequent.label))
                c_out[consequent.label] = FuzzyController.variables_default_val[consequent.label]
                is_exception = True

        if verbose>Constants.VERBOSE_BASIC:
            print("assessor inputs")
            print(self.controlsystem.input)
            print("rules activations after compute")
            for r in rules_activations:
                print(str(r) + "\n\t\t--->" + str(rules_activations[r]))
            print("c_outs " + str(c_out))

        self.controlsystem._reset_simulation()
        return c_out, rules_activations, is_exception

    def computeOutput(self, inputs, verbose=False):
        self.feed_inputs(inputs)
        return self.compute(verbose)

    def addRules(self, rules):
        for r in rules:
            if not str(r) in self.rules_str:
                self.control.addrule(r)
                self.rules_str.append(str(r))
        self.refreshInputs()
        print("size new arulebase: " + str(len([i for i in self.control.rules])))

    def getRules(self):
        return self.control.rules

    def getAllRules(self):
        return self.control.rules.all_rules

    def printRules(self):
        for rule in self.control.rules:
            print(rule)


class CreativeFuzzyController(FuzzyController):
    possible_inputs = []
    possible_outputs = []
    inputs_mf = {}
    outputs_mf = {}
    default_crisp_inputs = {}

    # note_ the following is a construct used to facilitate the interaction with GA. it is the concatenation of
    # all possible membership function values of the outputs
    concat_output_membership_functions = []

    @staticmethod
    def cleanClassVar():
        # del CreativeFuzzyController.possible_inputs
        # del CreativeFuzzyController.possible_outputs
        # del CreativeFuzzyController.inputs_mf
        # del CreativeFuzzyController.outputs_mf
        # del CreativeFuzzyController.default_crisp_inputs
        # del CreativeFuzzyController.concat_output_membership_functions
        # gc.collect()
        CreativeFuzzyController.possible_inputs = []
        CreativeFuzzyController.possible_outputs = []
        CreativeFuzzyController.inputs_mf = {}
        CreativeFuzzyController.outputs_mf = {}
        CreativeFuzzyController.default_crisp_inputs = {}
        CreativeFuzzyController.concat_output_membership_functions = []

    @staticmethod
    def initCreativeFuzzyControllerStaticInfo(possible_inputs, possible_outputs):
        CreativeFuzzyController.cleanClassVar()
        CreativeFuzzyController.possible_inputs = possible_inputs
        CreativeFuzzyController.possible_outputs = possible_outputs
        for i in possible_inputs:
            # print(i)
            # print(variables_universe[i])
            CreativeFuzzyController.inputs_mf[i] = ctrl.Antecedent(FuzzyController.variables_universe[i], i)
            for v in FuzzyController.dimensions_values[i]:
                # if v=="disabled": v = "disabled_left" a_inputs[i][v] = fuzzymath.fuzzy_or(variables_universe[i],
                # fuzz.trapmf(variables_universe[i], fuzzysets_values[i]['disabled_left']), variables_universe[i],
                # fuzz.trapmf(variables_universe[i], fuzzysets_values[i]['disabled_right']))[1] else: print(
                # fuzzysets_values[i][v])
                CreativeFuzzyController.inputs_mf[i][v] = fuzz.trapmf(FuzzyController.variables_universe[i], FuzzyController.fuzzysets_values[i][v])
        for a in possible_outputs:
            CreativeFuzzyController.outputs_mf[a] = ctrl.Consequent(FuzzyController.variables_universe[a], a)
            for v in FuzzyController.dimensions_values[a]:
                # print("creating a triangular membership function for variable "+str(a)+" value "+str(v))
                # print("the universe is "+str(variables_universe[a]))
                # print("the possible values are "+str(fuzzysets_values[a][v]))
                CreativeFuzzyController.outputs_mf[a][v] = fuzz.trapmf(CreativeFuzzyController.outputs_mf[a].universe, FuzzyController.fuzzysets_values[a][v])
            #     print(a_outputs[a][v])
            #     print(a_outputs[a][v].mf)
            # a_outputs[a].view()

        for a in possible_outputs:
            CreativeFuzzyController.concat_output_membership_functions = CreativeFuzzyController.concat_output_membership_functions+FuzzyController.dimensions_values[a]

    @staticmethod
    def viewRules():
        print("input")
        for x in CreativeFuzzyController.inputs_mf:
            CreativeFuzzyController.inputs_mf[x].view()
        print("output")
        for x in CreativeFuzzyController.outputs_mf:
            CreativeFuzzyController.outputs_mf[x].view()

    def __init__(self, rulebase):
        super().__init__(rulebase)


class AssessorFuzzyController(FuzzyController):
    possible_inputs = []
    possible_outputs = []
    inputs_mf = {}
    outputs_mf = {}
    default_crisp_inputs = {}

    @staticmethod
    def cleanClassVar():
        # del AssessorFuzzyController.possible_inputs
        # del AssessorFuzzyController.possible_outputs
        # del AssessorFuzzyController.inputs_mf
        # del AssessorFuzzyController.outputs_mf
        # del AssessorFuzzyController.default_crisp_inputs
        # gc.collect()
        AssessorFuzzyController.possible_inputs = []
        AssessorFuzzyController.possible_outputs = []
        AssessorFuzzyController.inputs_mf = {}
        AssessorFuzzyController.outputs_mf = {}
        AssessorFuzzyController.default_crisp_inputs = {}

    @staticmethod
    def initAssessorFuzzyControllerStaticInfo(possible_inputs, possible_outputs):
        AssessorFuzzyController.cleanClassVar()
        AssessorFuzzyController.possible_inputs = possible_inputs
        AssessorFuzzyController.possible_outputs = possible_outputs
        for i in possible_inputs:
            # print(i)
            # print(variables_universe[i])
            AssessorFuzzyController.inputs_mf[i] = ctrl.Antecedent(FuzzyController.variables_universe[i], i)
            for v in FuzzyController.dimensions_values[i]:
                # if v=="disabled": v = "disabled_left" a_inputs[i][v] = fuzzymath.fuzzy_or(variables_universe[i],
                # fuzz.trapmf(variables_universe[i], fuzzysets_values[i]['disabled_left']), variables_universe[i],
                # fuzz.trapmf(variables_universe[i], fuzzysets_values[i]['disabled_right']))[1] else: print(
                # fuzzysets_values[i][v])
                AssessorFuzzyController.inputs_mf[i][v] = fuzz.trapmf(FuzzyController.variables_universe[i],
                                                                      FuzzyController.fuzzysets_values[i][v])
        for a in possible_outputs:
            AssessorFuzzyController.outputs_mf[a] = ctrl.Consequent(FuzzyController.variables_universe[a], a)
            for v in FuzzyController.dimensions_values[a]:
                # print("creating a triangular membership function for variable "+str(a)+" value "+str(v))
                # print("the universe is "+str(variables_universe[a]))
                # print("the possible values are "+str(fuzzysets_values[a][v]))
                AssessorFuzzyController.outputs_mf[a][v] = fuzz.trapmf(AssessorFuzzyController.outputs_mf[a].universe,
                                                                       FuzzyController.fuzzysets_values[a][v])
            #     print(a_outputs[a][v])
            #     print(a_outputs[a][v].mf)
            # a_outputs[a].view()

    @staticmethod
    def viewRules():
        print("input")
        for x in AssessorFuzzyController.inputs_mf:
            AssessorFuzzyController.inputs_mf[x].view()
        print("output")
        for x in AssessorFuzzyController.outputs_mf:
            AssessorFuzzyController.outputs_mf[x].view()

    def __init__(self, rulebase):
        super().__init__(rulebase)
