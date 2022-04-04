from statistics import mean

from mesa import Agent

import utils.constants as Constants
from simulation.agents.utils.fuzzycontroller import FuzzyController
from simulation.agents.utils.utils import importDataFromFiles, sample_inputs, getSimpleModalityLinguisticInterpretation


class FuzzyPatientAgent(Agent):
    """ A patient agent that reacts based on preferences retrieved from an excel file and modeled as fuzzy rules of the type:
    IF x_1 AND ... AND x_m then FEEDBACK is high/low/medium/very hihg/very low
    """

    def __init__(self, unique_id, model, env, sim_param, ta_details, variables_details, pref_details, verbose):
        super().__init__(unique_id, model)
        self.name = unique_id
        self.env = env
        self.proposedActivity = []
        self.feedback = 0.0
        self.lastResponse = ""
        # self.preferences = preferences
        self.actions_centers = ta_details  # todo i'm assuming for now they are the same as those given to NAO. this is a dataframe like <action, lval_1, lval_2, lval_3, c_lval_1, c_lval_2, c_lval_3>
        self.verbose = verbose

        """ Importing all initialization details from input files """

        self.contextual_var, self.person_var, self.eval_var, self.variables_default_val, self.var_possible_val, \
        self.ta, self.weights_therapeutic_interventions, self.action_var, self.actions_to_ti, self.ta_to_actions, self.variables_universe, \
        self.dimensions_values, self.fuzzysets_values, self.range_step, self.info_rules_to_add \
            = importDataFromFiles(ta_details, variables_details, sim_param, preferences_details=pref_details)

        """ Initializing the Controllers (one per type fo activit, i.e. therapyy) """
        self.controllers = {}
        self.controllers_inputs_mf = {}
        self.controllers_outputs_mf = {}
        for ta in self.ta:
            controller_possible_inputs = self.contextual_var + self.person_var + self.ta_to_actions[ta]
            # todo assumes that every controller has same output var
            controller_possible_outputs = self.eval_var
            self.controllers[ta] = FuzzyController(self.dimensions_values, self.variables_universe,
                                                 self.fuzzysets_values, self.variables_default_val,
                                                 controller_possible_inputs, controller_possible_outputs,
                                                 [])
            controller_inputs_mf, controller_outputs_mf = FuzzyController.getMF(self.dimensions_values,
                                                                            self.variables_universe,
                                                                            self.fuzzysets_values,
                                                                            self.variables_default_val,
                                                                            controller_possible_inputs,
                                                                            controller_possible_outputs)
            self.controllers_inputs_mf[ta] = controller_inputs_mf
            self.controllers_outputs_mf[ta] = controller_outputs_mf

        for ir in self.info_rules_to_add:
            ta = self.actions_to_ti[ir[1][0][0]]
            ar = self.controllers[ta].createAndAddRule(ir[0]+ir[1], ir[2], self.controllers_inputs_mf[ta], self.controllers_outputs_mf[ta])

        if self.verbose == Constants.VERBOSE_BASIC:
            print("rules of the fuzzy patient")
            for c in self.controllers:
                print(c)
                print(self.controllers[c].getAllRules())

        self.curr_interaction = 0

    def step(self):
        """
        Step function executed at every simuation step
        The patient
        (i) gets the current input from the environment agent
         (ii) receives the suggested activity from the SAR
         (iii) feeds (i) and (ii) to the fuzzy controller to determine feedback output
         (iv) sets its current feedback to the value obtained in (iii). This value will be retrieved then by the SAR
         during the next step.
        """
        self.update_crispval(self.env.context)  # update the context
        if self.proposedActivity[0] == "question":
            if self.verbose == Constants.VERBOSE_BASIC:
                print("ok sure! let me tell you something...")
            self.feedback = 5
            self.lastResponse = "yes"
        else:
            curr_input = sample_inputs(False, 0, self.curr_interaction, self.variables_default_val, self.action_var,
                                       self.fuzzysets_values, self.variables_universe)
            prop_activity = self.proposedActivity[0]
            prop_activity_modality = self.proposedActivity[1]

            inputs = dict(curr_input)  # I create a copy fo the dict
            inputs[prop_activity] = prop_activity_modality

            ta = self.actions_to_ti[prop_activity]
            self.controllers[ta].feed_inputs(inputs)
            is_exception = False
            controllerout = []
            try:
                c_out, a_rules_activations, is_exception = self.controllers[ta].compute(verbose=False)
                controllerout = [c_out[co] for co in c_out]
            except:
                is_exception = True
                #todo assumes that every controller has same output var
                for v in self.eval_var:
                    controllerout.append(self.variables_default_val[v])

            if len(controllerout)==0:
                for v in self.eval_var:
                    controllerout.append(self.variables_default_val[v])

            self.feedback = mean(controllerout)

            prop_activity_interpretation = getSimpleModalityLinguisticInterpretation(self.actions_centers, prop_activity, prop_activity_modality)

            if self.feedback >= 5:
                if self.verbose == Constants.VERBOSE_BASIC:
                    print("ok sure! let's " + str(prop_activity_interpretation) + " " + str(prop_activity))
                self.lastResponse = "yes"
            else:
                if self.verbose == Constants.VERBOSE_BASIC:
                    print("I don't really feel like " + str(prop_activity_interpretation) + " " + str(
                        prop_activity) + "...")
                self.lastResponse = "no"

        if self.verbose == Constants.VERBOSE_BASIC:
            print("feedback: " + str(self.feedback))

    def prepareForNewInteraction(self):
        """
        Function called after every interaction to reset the interaction-related variables
        """
        self.proposedActivity = []
        self.feedback = 0.0
        self.lastResponse = ""
        self.curr_interaction = self.curr_interaction + 1


    def update_crispval(self, val_dict):
        """
        Updates the values of the input variables according to the given values (retrieved from the environment)
        """
        for v in val_dict:
            if v in self.variables_default_val:
                self.variables_default_val[v] = val_dict[v]



