import time

from simulation.agents.utils.fuzzycontroller import FuzzyController
from simulation.agents.sar.gafc import GAFC
from simulation.agents.utils.utils import getDisabledCounterRule, getDisabledCounterRuleString

import skfuzzy as fuzz
from skfuzzy import control as ctrl

import utils.constants as Constants


class Personalizer:
    """
    Class of the Creative Personalization module of the SAR agent. The Peronalizer is the module that takes care of
    evolving the rules of the creative controller of the SAR via genetic algorithms by accounting for the experience
    acquired interacting with the patient.
    The Personalizer makes use of the GAFC interface to interface the creative controller of the SAR with the genetic algo.
    Additionally the Personalizer takes care of the update of the credits.
    Optionally, the personalizer can also be used to update the weights assigned to the different types of activities,
    and to expand the rule base of the assessor controller [these two options, however, are currently not fully supported].
    """

    def __init__(self, nao):
        self.nao = nao
        self.param = self.nao.param
        self.verbose = self.nao.verbose
        self.gafc = GAFC(self.param, self.nao)

    def updateCredits(self, collectedKnowledge, rules_credits, variables_default_val):
        """
        Function used to update the credits of the rules based on the collected knowledge (the interactions memory)
        Credits are updated based on the rule:
        c_{r,t+1} \gets \credit + \learningrate\cdot \payoff\cdot\frac{\rstr}{\sum_{r'\in \rules}s_{r',t}}
        where, \rstr~ denotes the firing strength of a rule $r$ at time $t$,
        and $\learningrate\in[0,1]$ is a constant representing the learning rate.
        """
        new_credits = dict(rules_credits) #creating a copy so I modify the copy
        for k in collectedKnowledge:  # one per each step
            sum_act = 0
            u = 0
            for r in k['rules_activations']:
                sum_act = sum_act + k['rules_activations'][r]
            if sum_act > 0 and len(k["sugg"])>0:  # serves to check if any suggestion was given but also to avoid dividing by 0
                for r in k['rules_activations']:
                    if (k["sugg"][0] in r) and (k['rules_activations'][r] > 0):
                        num=9.0
                        den=5.0
                        # num/det = 1.8
                        if (k['detected_emotions'] - variables_default_val['emotions']) >= 0:
                            reward = pow(k['detected_emotions'] - variables_default_val['emotions'], num/den)
                        else:
                            reward = pow(-1, num) * pow(abs(k['detected_emotions'] - variables_default_val['emotions']), num / den)
                        change = self.param["learning_rate_credits"] * (reward * (k['rules_activations'][r] / sum_act))
                        new_credits[r] = new_credits[r] + change  # the update rule
                        # I update conversely a disabled rule if the option is enabled
                        #todo this is still under development
                        if self.param["compute_disabled_counterparts_credits"]:
                            disabled_rule_str = getDisabledCounterRule(r)
                            new_credits[disabled_rule_str] = new_credits[
                                                                        disabled_rule_str] - change  # the update rule
                        u = u + 1
                if self.verbose == Constants.VERBOSE_BASIC:
                    print("updated credits of " + str(u) + " rules")
        return new_credits

    def run(self):
        """ Function that executes the creative personalization phase.
        It is called in between different interactions between the SAR and the Patient.
        It mainly performs two activities:
        1. Updates the credits of the rules based on the knowledge collected in the previous interactions
        2. It evolves the c-rules

        Optionally it can also perform (even though currently not supported)
        - Reassessment of the weights associated to the different types of activities
        - Update of the a-rules based on the acquired knowledge"""

        if self.param["personalize"]:
            """ [Optional, currently not supported] Reassess the weights of the types of therapies"""
            if self.param["reassess_weights"]:
                if self.verbose==Constants.VERBOSE_BASIC:
                    print("Reassessing weights ...")
                start_time = time.time()
                self.nao.weights_therapeutic_interventions = self.updateWeights(self.nao.collectedKnowledge,
                                                                                         self.nao.weights_therapeutic_interventions,
                                                                                         self.nao.actions_to_ti)
                if self.verbose == Constants.VERBOSE_BASIC:
                    print("UPDATE WEIGHTS TIME ELAPSED --- %s seconds ---" % (time.time() - start_time))

            """ Update the credits """
            if self.verbose == Constants.VERBOSE_BASIC:
                print("Updating rules credits ...")
            self.nao.rules_credits = self.updateCredits(self.nao.collectedKnowledge, self.nao.rules_credits,
                                                                 self.nao.variables_default_val)

            """ [Optional, currently not supported] Updates the a-rules"""
            if self.param["update_arules"]:
                if self.verbose == Constants.VERBOSE_BASIC:
                    print("Expanding a-rules ...")
                start_time = time.time()
                rules_to_add = self.expandARules(self.nao.collectedKnowledge, self.nao.assessors_inputs_mf,
                                                              self.nao.assessors_outputs_mf)
                for a in rules_to_add: #todo this is currently not supported and needs to be fixed (for now it is assumed that an empty list is returned so the loop has no effect)
                    new_rule_base = self.nao.assessors[a].addRules(rules_to_add[a])
                if self.verbose>Constants.VERBOSE_BASIC:
                    print("NEW A-RULES:")
                    for a in self.nao.assessors:
                        self.nao.assessors[a].printRules()
                if self.verbose == Constants.VERBOSE_BASIC:
                    print("UPDATE A-RULES TIME ELAPSED --- %s seconds ---" % (time.time() - start_time))

            """ Evolve the C-Rules with Genetic Algorithms"""
            if self.verbose == Constants.VERBOSE_BASIC:
                print("Evolving c-rules ...")
            start_time = time.time()
            creative_rulebase, population = self.evolveCRules(self.nao.last_population)
            self.nao.creative_controller = FuzzyController(self.nao.dimensions_values,
                                                          self.nao.variables_universe,
                                                          self.nao.fuzzysets_values,
                                                          self.nao.variables_default_val,
                                                          self.nao.creative_controller_possible_inputs,
                                                          self.nao.creative_controller_possible_outputs,
                                                          creative_rulebase)
            if self.verbose == Constants.VERBOSE_BASIC:
                print(str(len(creative_rulebase)) + " c-rules")
            self.nao.last_population = population
            self.insertNewCredits()
            if self.verbose == Constants.VERBOSE_BASIC:
                print("EVOLVE C-RULES TIME ELAPSED --- %s seconds ---" % (time.time() - start_time))

    def evolveCRules(self, init_population):
        """ Function that invokes the genetic algorithm to evolve the c-rules and returns a new population (rule  base)
        to replace the current c-rule base
        """
        return self.gafc.run(init_population)

    def insertNewCredits(self, cr=None):
        """ Function to update the credits of the rules.
        It is useful after rules never tried before are introduced in the rule base.
        In this case a default credit (or a credit given in input @cr) is associated and stored for the new rules """
        for r in self.nao.creative_controller.getRules():
            creative_rule_str = str(r)
            try:
                cc = self.nao.rules_credits[creative_rule_str]
            except:
                credit = self.param["default_credit"]
                if not cr is None:
                    credit = cr
                self.nao.rules_credits[creative_rule_str] = credit

            if self.param["compute_disabled_counterparts_credits"]:
                # I also add a counterpart for the disabled rule
                disabled_rule_str = getDisabledCounterRuleString(r)
                if (not disabled_rule_str is None) and (not disabled_rule_str in self.nao.rules_credits):
                    self.nao.rules_credits[disabled_rule_str] = self.param["default_credit"]



    def expandARules(self, collectedKnowledge, assessor_inputs_mf, assessor_outputs_mf):
        print("Warning: TODO this function needs to be fixed. The function currently does not do anything")
        # todo
        """ note_ need to fix also  the line right after the call of this function, 
        #  because we need to add rules to the apprppriate assessors. so maybe rules_to_add should be a dictionary instead of a list
        """
        rules_to_add = []
        """ the following part has been commented. It is correct but needs to be fixed to be compatible with latest changes.
        for k in collectedKnowledge:
            if len(k['sugg']) > 0:
                ant_crisp = {}
                ant_crisp.update(k["context"])
                ant_crisp.update({k["sugg"][0]: k["sugg"][1]})
                cons_crisp = {"emotions": k["emotions"]}
                print("cons crisp: "+str(cons_crisp))

                rule_ant = {}
                rule_cons = {}
                for var in ant_crisp:
                    val = ant_crisp[var]
                    max_lingval_membership = -100
                    best_lingval = None
                    for ling_val in FuzzyController.dimensions_values[var]:
                        lingval_mem = fuzz.interp_membership(FuzzyController.variables_universe[var],
                                                             assessor_inputs_mf[var][ling_val].mf, val)
                        if lingval_mem > max_lingval_membership:
                            max_lingval_membership = lingval_mem
                            best_lingval = ling_val
                    rule_ant[var] = best_lingval

                for var in cons_crisp:
                    val = cons_crisp[var]
                    max_lingval_membership = -100
                    best_lingval = None
                    for ling_val in FuzzyController.dimensions_values[var]:
                        lingval_mem = fuzz.interp_membership(FuzzyController.variables_universe[var],
                                                             assessor_outputs_mf[var][ling_val].mf, val)
                        # print(str(ling_val)+" -> "+str(lingval_mem))
                        # print(self.variables_universe[var])
                        # print(self.assessor_outmf[var][ling_val].view())
                        if lingval_mem > max_lingval_membership:
                            max_lingval_membership = lingval_mem
                            best_lingval = ling_val
                    rule_cons[var] = best_lingval

                ante = None
                for a in rule_ant:
                    ling_var = assessor_inputs_mf[a]
                    ling_val = rule_ant[a]
                    term = ling_var[ling_val]
                    if ante is None:
                        ante = term
                    else:
                        ante = ante & term
                conseq = None
                for c in rule_cons:
                    ling_var = assessor_outputs_mf[c]
                    ling_val = rule_cons[c]
                    term = ling_var[ling_val]
                    if conseq is None:
                        conseq = term
                    else:
                        conseq = conseq & term

                rule = ctrl.Rule(ante, conseq)
                print("Acquired knowledge for new rule: " + str(rule))
                rules_to_add.append(rule)

        return rules_to_add
        """
        return rules_to_add

    def updateWeights(self, collectedKnowledge, curr_weights, actions_to_ti):
        """ Function to be used to update the weights of the different types of activities based on the
        collected knowledge (the interactions memory) """
        print("Warning: TODO this function needs to be fixed. Currently it uses a simple rule, which was not fully tested")
        new_weights = {}
        for k in collectedKnowledge:
            if len(k['sugg']) > 0:
                action = k["sugg"][0]
                w_ta = curr_weights[actions_to_ti[action]]
                new_weights[
                    actions_to_ti[action]] = w_ta - self.param["learning_rate_weights"] * (
                            k['expected_emotions'] - k['detected_emotions'])
        return new_weights