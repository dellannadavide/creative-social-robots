# import cProfile
import random
import traceback
from statistics import *

from mesa import Agent

from simulation.agents.utils.fuzzycontroller import *
from simulation.agents.sar.sar_agent_personalizer import Personalizer
from simulation.agents.utils.utils import importDataFromFiles, sample_inputs, getRepetitionCost
import utils.constants as Constants
# from simulation.utils.utils import analyze_memory
from random import Random


class SARAgent(Agent):
    """ SAR agent implementing a SAR for creative long-term personalization.
    It combines fuzzy control with genetic algorithms to autonomously evolve at every interaction the
    fuzzy rule base that determines its suggestions in a given context.
    """

    def __init__(self, unique_id, model, patient, env, filename, param, ta_details, variables_details, verb,
                 pref_details=None):
        super().__init__(unique_id, model)
        self.filename = filename
        self.patient = patient
        self.env = env
        self.ta_details = ta_details
        self.verbose = verb
        self.param = param
        self.r = Random()
        self.r.seed(42)

        """ Importing all initialization details from input files """

        self.contextual_var, self.person_var, self.eval_var, self.variables_default_val, self.var_possible_val, \
        self.ta, self.weights_therapeutic_interventions, self.action_var, self.actions_to_ti, self.ta_to_actions, self.variables_universe, \
        self.dimensions_values, self.fuzzysets_values, self.range_step, self.info_rules_to_add \
            = importDataFromFiles(ta_details, variables_details, self.param, preferences_details=pref_details)

        """ Initializing the info for the FuzzyControllers """

        """ Initializing the Assessor"""
        self.assessors = {}
        self.assessors_possible_inputs = {}
        self.assessors_possible_outputs = {}
        self.assessors_inputs_mf = {}
        self.assessors_outputs_mf = {}
        for ta in self.ta:
            self.assessors_possible_inputs[ta] = self.contextual_var + self.person_var + self.ta_to_actions[ta]
            # todo assuming that every assessor controller has the same outputs
            self.assessors_possible_outputs[ta] = self.eval_var
            self.assessors[ta] = FuzzyController(self.dimensions_values, self.variables_universe,
                                                 self.fuzzysets_values, self.variables_default_val,
                                                 self.assessors_possible_inputs[ta],
                                                 self.assessors_possible_outputs[ta],
                                                 [])
            assessor_inputs_mf, assessor_outputs_mf = FuzzyController.getMF(self.dimensions_values,
                                                                            self.variables_universe,
                                                                            self.fuzzysets_values,
                                                                            self.variables_default_val,
                                                                            self.assessors_possible_inputs[ta],
                                                                            self.assessors_possible_outputs[ta])
            self.assessors_inputs_mf[ta] = assessor_inputs_mf
            self.assessors_outputs_mf[ta] = assessor_outputs_mf

        """ Initializing the Creative Controller """
        self.creative_controller_possible_inputs = self.contextual_var + self.person_var
        self.creative_controller_possible_outputs = self.action_var
        self.creative_controller = FuzzyController(self.dimensions_values, self.variables_universe,
                                                   self.fuzzysets_values, self.variables_default_val,
                                                   self.creative_controller_possible_inputs,
                                                   self.creative_controller_possible_outputs, [])
        self.creative_controller_inputs_mf, self.creative_controller_outputs_mf = FuzzyController.getMF(
            self.dimensions_values,
            self.variables_universe,
            self.fuzzysets_values,
            self.variables_default_val,
            self.creative_controller_possible_inputs,
            self.creative_controller_possible_outputs)
        self.concat_output_membership_functions = []
        for a in self.creative_controller_possible_outputs:
            self.concat_output_membership_functions = self.concat_output_membership_functions + \
                                                      self.dimensions_values[a]

        """ Add the initial rules, if any"""
        rules_info_for_initial_pop = []
        for ir in self.info_rules_to_add:
            if (not ir[2][0][1] == "poor") and (not ir[2][0][1] == "very poor"):
                cr = self.creative_controller.createAndAddRule(ir[0], ir[1], self.creative_controller_inputs_mf,
                                                               self.creative_controller_outputs_mf)
                rules_info_for_initial_pop.append(ir)
            ta = self.actions_to_ti[ir[1][0][0]]
            ar = self.assessors[ta].createAndAddRule(ir[0] + ir[1], ir[2], self.assessors_inputs_mf[ta],
                                                     self.assessors_outputs_mf[ta])

        self.curr_iter_suggestions = []
        self.last_suggestion = []
        self.collectedKnowledge = []
        self.last_context = {}
        self.lastGAmodel = None
        self.expected_feedback = 0
        self.last_rules_activations = {}
        self.logInteractions = []
        self.curr_interaction = 0
        self.rules_credits = {}
        self.last_population = None

        """ Create the Personalizer, for Creative Personalization """
        self.personalizer = Personalizer(self)
        self.last_population = self.personalizer.gafc.rulebase2population(rules_info_for_initial_pop)
        self.personalizer.insertNewCredits(2)
        if self.last_population is None:
            self.personalizer.run()

        if self.verbose == Constants.VERBOSE_BASIC:
            print("size initial population from rules")
            print(len(self.last_population))
            print("printing the rules of sar")
            print("creative rules")
            print(self.creative_controller.getAllRules())
            print("assessor rules")
            for assessor in self.assessors:
                print(self.assessors[assessor].getAllRules())

    def prepareForNewInteraction(self):
        self.curr_iter_suggestions = []
        self.last_suggestion = []
        self.last_context = {}
        for k in self.collectedKnowledge:
            self.logInteractions.append(k)
        self.collectedKnowledge.clear()
        self.lastGAmodel = None
        self.expected_feedback = 0
        self.last_rules_activations = {}
        self.curr_interaction = self.curr_interaction + 1

    def step(self):
        """ Step function executed at every simuation step """

        """ First updates the variables values of the current time form the environment """
        self.update_crispval(self.env.context)

        """
        here the decision making of the agent
        to determine which activity to suggest to the patient
        i apply the creative controller to the current context
        """
        curr_input = sample_inputs(False, 0, self.curr_interaction, self.variables_default_val, self.action_var,
                                   self.fuzzysets_values, self.variables_universe)
        c_out, rules_activations = self.creative_controller.computeOutput(curr_input, False)

        """ i obtain a number of ouput crisp values.
        i determine which one achieves the max expected output w.r.t. the a-rules """
        best_a = None
        best_a_val = -1000
        best_a_exphapp = 5
        if self.verbose > Constants.VERBOSE_BASIC:
            print("rules activations")
            for a in rules_activations:
                if rules_activations[a] > 0:
                    print(str(a) + "\n\t\t\t-> " + str(rules_activations[a]))
        for item in c_out.items():  # for each pair <activity, crisp output>
            if self.verbose > Constants.VERBOSE_BASIC:
                print(item)
            if not item[
                       0] in self.curr_iter_suggestions:  # if i didn't suggest the same activity already in the same interaction
                inputs = dict(curr_input)  # I create a copy fo the dict
                inputs[item[0]] = item[1]
                assessor_id = self.actions_to_ti[item[0]]
                self.assessors[assessor_id].feed_inputs(inputs)
                assout = []
                try:
                    a_out, a_rules_activations = self.assessors[assessor_id].compute(verbose=False)
                    assout = [a_out[ao] for ao in a_out]
                except:
                    traceback.print_exc()
                    # todo the following assumes that every assessor controller has same eval var
                    for v in self.eval_var:
                        assout.append(self.variables_default_val[v])
                if len(assout) == 0:
                    for v in self.eval_var:
                        assout.append(self.variables_default_val[v])
                w_ta = self.weights_therapeutic_interventions[self.actions_to_ti[item[0]]]

                avg_credit_rules_that_suggested_action = 1.0
                nr_rules_that_suggested_action = 0
                for r in rules_activations:
                    if (rules_activations[r] > 0) and (str(item[0]) in str(r)):
                        avg_credit_rules_that_suggested_action = avg_credit_rules_that_suggested_action + \
                                                                 self.rules_credits[str(r)]
                        nr_rules_that_suggested_action = nr_rules_that_suggested_action + 1
                if nr_rules_that_suggested_action > 0:
                    avg_credit_rules_that_suggested_action = (
                                                                         avg_credit_rules_that_suggested_action - 1.0) / nr_rules_that_suggested_action
                repetition_cost = 1.0
                a_val = (mean(assout) * w_ta * avg_credit_rules_that_suggested_action) / repetition_cost
                if (a_val > best_a_val) and (
                        item[1] >= (self.variables_default_val[item[0]] + self.range_step[item[0]])):
                    best_a = item
                    best_a_val = a_val
                    best_a_exphapp = mean(assout)

        """ I suggest the activity with best expected outcome and store the information to populate the interactions memory"""
        self.proposeActivity(best_a)
        if not best_a is None:
            if (self.verbose > Constants.VERBOSE_FALSE) and (self.verbose <= Constants.VERBOSE_BASIC):
                print("proposing activity" + str(best_a) + " which has expected feedback: " + str(
                    best_a_exphapp) + ", which weighted is " + str(best_a_val))
            self.curr_iter_suggestions.append(best_a[0])
            self.last_suggestion = best_a
        else:
            if (self.verbose > Constants.VERBOSE_FALSE) and (self.verbose <= Constants.VERBOSE_BASIC):
                print("the activity proposed is " + str(
                    best_a) + " so I don't suggest anything. I will ask a question instead")
            self.last_suggestion = []
        self.expected_feedback = best_a_exphapp
        self.last_context = self.env.context.copy()
        self.last_rules_activations = rules_activations

    def collectKnowledge(self):
        """ Function to collect the information from the current step into a knowledge base (the interactions memory)
        that will be used for the creative personalization, and for loggin purposes """
        self.collectedKnowledge.append({"context": self.last_context,
                                        "sugg": self.last_suggestion,
                                        "expected_emotions": self.expected_feedback,
                                        "rules_activations": self.last_rules_activations,
                                        "response": self.patient.lastResponse,
                                        "emotions": self.patient.feedback,
                                        "detected_emotions": self.detectPatientFeedback(),
                                        "rulebase": self.creative_controller.getAllRules()})

    def detectPatientFeedback(self):
        """ Function to collect feedback from the Patient Agent and possibly to simulate the presence of noise in the detection.
        If no noise is applied, the exact feedback provided by the Patient Agent is retrived.
        Otherwise the feedback is retrieved and then transformed based on the noise."""

        noise_prob = self.param["noise_probability"]
        resp_with_noise = self.patient.feedback
        if noise_prob > 0:
            max_noise_val = 10  # todo hardcoded for the noise
            min_noise_val = -10
            max_eval = 10
            min_eval = 0
            for v in self.eval_var:
                max_eval = self.variables_universe[v][-1]
                min_eval = self.variables_universe[v][0]
                break  # todo assuming that every eval var is the same

            actual_resp = self.patient.feedback
            prob = self.r.uniform(0, 1)
            if prob <= noise_prob:
                noise = 0.0
                if self.param["noise_type"] == "gaussian":
                    noise = self.r.gauss(0, 2)
                    if noise > max_noise_val:
                        noise = max_noise_val
                    if noise < min_noise_val:
                        noise = min_noise_val
                if self.param["noise_type"] == "inv_gaussian":
                    noise = self.r.gauss(0, 2)
                    if noise > max_noise_val:
                        noise = max_noise_val
                    if noise < min_noise_val:
                        noise = min_noise_val
                    if noise < 0:
                        noise = (min_noise_val - max_noise_val) - noise
                    elif noise > 0:
                        noise = max_noise_val - noise
                    else:
                        if self.r.uniform(0, 1) > 0.5:
                            noise = max_noise_val
                        else:
                            noise = min_noise_val
                if self.param["noise_type"] == "reversed_feedback":
                    dist_from_max = max_eval - actual_resp
                    noise = min_eval + dist_from_max - actual_resp  # i remove actual_resp to obtain the noise, otherwise it's already the new response, I add it again later
                resp_with_noise = actual_resp + noise
                if resp_with_noise > max_eval:
                    resp_with_noise = max_eval
                if resp_with_noise < min_eval:
                    resp_with_noise = min_eval
        return resp_with_noise

    def proposeActivity(self, activity):
        """ Function that suggests an activity to the patient.
        Practically this function is simply used to communicate the suggestion to the patient.
        In case the input @activity is None, the SAR asks a question (for now simply a default 'question')
        """
        a = activity
        if (activity is None):
            a = ["question", 0]
        self.patient.proposedActivity = a
        if self.verbose == Constants.VERBOSE_BASIC:
            if (activity is None):
                print("tell me something about you! How do you feel today?")
            elif self.patient.lastResponse == 'no':
                print("Oh ok, how about " + a[0] + "?")
            elif self.patient.lastResponse == 'yes':
                print("Great! That was fun! " + a[0])
            else:
                print("Hey " + self.patient.name + ", would you like to " + a[0])
            print(self.patient.proposedActivity)
        return a

    # def evalInteraction(self):
    #     """ Function used to """
    #     detected_emotions = []
    #     for k in self.collectedKnowledge:
    #         detected_emotions.append(k["emotions"])
    #     avg_feedback = mean(detected_emotions)
    #     if self.verbose == Constants.VERBOSE_BASIC:
    #         print("Average feedback detected:" + str(avg_feedback))
    #     return avg_feedback

    def update_crispval(self, val_dict):
        """
        Updates the values of the input variables according to the given values (possibly retrieved from the environment)
        """
        for v in val_dict:
            if v in self.variables_default_val:
                self.variables_default_val[v] = val_dict[v]
