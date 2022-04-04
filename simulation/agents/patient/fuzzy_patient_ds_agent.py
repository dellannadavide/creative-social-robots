import random
from statistics import mean

from mesa import Agent

import utils.constants as Constants
from simulation.agents.utils.fuzzycontroller import FuzzyController
from simulation.agents.utils.utils import importDataFromFiles, sample_inputs, getSimpleModalityLinguisticInterpretation, \
    getSimpleInputLinguisticInterpretation, getActionCenterCrispVal, getInputLinguisticInterpretation

import csv


class FuzzyPatientDSAgent(Agent):
    """
    A fuzzy patient generated based on the preferences elicited from humans via a survey about preferred daily activities
    """

    def __init__(self, unique_id, model, env, sim_param, ta_details, variables_details, pref_details, verbose, memory_info={}):
        super().__init__(unique_id, model)
        self.name = unique_id
        self.env = env
        self.proposedActivity = []
        self.feedback = 0.0
        self.lastResponse = ""
        # self.preferences = preferences
        self.actions_centers = ta_details  # todo i'm assuming for now they are the same as those given to NAO. this is a dataframe like <action, lval_1, lval_2, lval_3, c_lval_1, c_lval_2, c_lval_3>
        self.verbose = verbose
        self.variables_details = variables_details
        self.memory_info = memory_info

        self.memory = []
        self.feedback_dict = {}

        self.a = 0
        self.b = 14
        self.c = 14
        self.mem_size = 14
        self.feedback_change = 0
        self.mood_swing_freq = 0.0
        self.mood = 1
        if (self.mood_swing_freq > 0):
            if self.r.uniform(0, 1) >= 0.5:
                self.mood = -1

        if len(self.memory_info)>0:
            self.a = memory_info["a"]
            self.b = memory_info["b"]
            self.c = memory_info["c"]
            self.mem_size = memory_info["mem_size"]
            self.feedback_change = memory_info["feedback_change"]
            self.mood_swing_freq = memory_info["mood_swing_freq"]
            self.mood = 1
            if (self.mood_swing_freq > 0):
                if self.r.uniform(0, 1) >= 0.5:
                    self.mood = -1


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
            ar = self.controllers[ta].createAndAddRule(ir[0] + ir[1], ir[2], self.controllers_inputs_mf[ta],
                                                       self.controllers_outputs_mf[ta])

        self.curr_interaction = 0

    def step(self):
        """
        Step function executed at every simuation step
        The patient
        (i) gets the current input from the environment agent
         (ii) for every possible activity a in its knowledge base
         (iii) feeds (i) and a to the fuzzy controller to determine feedback output
         (iv) returns a tuple context-activity-feedback, which will be used to generate a dataset
        """

        # choosen_key = self.getPreferredActivityInContext(self.env.context)
        #
        # # add it to the info dataset
        # if not choosen_key in self.dataset_actions_ids:
        #     self.dataset_actions_ids[choosen_key] = self.next_dataset_action_id
        #     self.next_dataset_action_id = self.next_dataset_action_id +1
        #     self.writer_info.writerow([self.dataset_actions_ids[choosen_key], choosen_key])
        # action_id = self.dataset_actions_ids[choosen_key]
        #
        # datapoint = []
        # for c in self.env.context:
        #     datapoint = datapoint + [self.env.context[c]]
        # datapoint = datapoint + [action_id]# [max_key, feedback_activities[max_key]]
        # # print(datapoint)
        # self.dataset.append(datapoint)
        # self.writer.writerow(datapoint)

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

    def generateDatasetPreferredActivities(self, size, dataset_name, dataset_info_name):
        dataset = []
        dataset_actions_ids = {}

        next_dataset_action_id = 1
        f = open(dataset_name, "w", newline="")
        writer = csv.writer(f)
        f_info = open(dataset_info_name, "w", newline="")
        writer_info = csv.writer(f_info)

        for i in range(size):
            print(i, end='...')
            context = self.env.getRandomEnvVal()
            choosen_key = self.getPreferredActivityInContext(context)

            # add it to the info dataset
            if not choosen_key in dataset_actions_ids:
                dataset_actions_ids[choosen_key] = next_dataset_action_id
                next_dataset_action_id = next_dataset_action_id + 1
                writer_info.writerow([dataset_actions_ids[choosen_key], choosen_key])
            action_id = dataset_actions_ids[choosen_key]

            datapoint = []
            for c in context:
                datapoint = datapoint + [context[c]]
            datapoint = datapoint + [action_id]  # [max_key, feedback_activities[max_key]]
            # print(datapoint)
            dataset.append(datapoint)
            writer.writerow(datapoint)

    def getPreferredActivityInContext(self, context):
        feedback_activities = {}
        for ta in self.ta:
            for a in self.ta_to_actions[ta]:
                prop_activity = a
                prop_activity_modality = getActionCenterCrispVal(self.actions_centers,
                                                                 a)  # self.findMiddle(self.variables_universe[a])
                feedback_activities[a] = self.getFeedbackForActivityInContext(context, ta, prop_activity,
                                                                              prop_activity_modality)

        max_key = max(feedback_activities, key=feedback_activities.get)
        max_keys = []
        positive_keys = []
        for k in feedback_activities:
            if feedback_activities[k] == feedback_activities[max_key]:
                max_keys.append(k)
            if feedback_activities[k] > 5.0:
                positive_keys.append(k)
        if random.uniform(0,1)<=0.8: #in 80% of cases I pick one of the best, in the remaining 20% I pick one random among the positive
            choosen_key = random.choice(max_keys)
        else:
            choosen_key = random.choice(positive_keys)
        return choosen_key

    def getFeedbackForActivityInContext(self, context, therapy_activity, activity, activity_modality,default_feedback=-1):
        self.update_crispval(context)
        curr_input = sample_inputs(False, 0, self.curr_interaction, self.variables_default_val, self.action_var,
                                   self.fuzzysets_values, self.variables_universe)
        context_interpr = getInputLinguisticInterpretation(
                self.variables_details, context, self.var_possible_val,
                self.variables_universe, self.dimensions_values)
        inputs = dict(curr_input)  # I create a copy fo the dict
        inputs[activity] = activity_modality
        # ta = self.actions_to_ti[prop_activity]
        # print(activity)
        # print(inputs)
        self.controllers[therapy_activity].feed_inputs(inputs)
        is_exception = False
        controllerout = []
        try:
            c_out, a_rules_activations, is_exception = self.controllers[therapy_activity].compute(verbose=False)
            controllerout = [c_out[co] for co in c_out]
        except:
            is_exception = True
            # todo assumes that every controller has same output var
            for v in self.eval_var:
                controllerout = []
                if default_feedback==-1:
                    controllerout.append(self.variables_default_val[v])
                else:
                    controllerout.append(0.0)

        if (len(controllerout) == 0) or is_exception:
            controllerout = []
            for v in self.eval_var:
                if default_feedback == -1:
                    controllerout.append(self.variables_default_val[v])
                else:
                    controllerout.append(0.0)
        # print(str(mean(controllerout)), end=' ')
        # print(c_out)
        # print(a_rules_activations)
        # print(controllerout)
        fis_feedback = mean(controllerout)

        if len(self.memory_info) > 0:
            index_str = str(context_interpr) + "--" + str(activity) + "-" + str(activity_modality)

            """ First time we are seeing the activity"""
            if not index_str in self.feedback_dict:
                    self.feedback_dict[index_str] = fis_feedback

            feedback_suggestion = self.feedback_dict[index_str]

            """ After some time I adjust the feedback"""
            already_suggested = self.memory.count(str(activity))
            # print("already suggested "+str(already_suggested)+" times")
            change = 0.0
            if already_suggested <= self.a:  # before reaching a repetition I keep increasing the feedback more and more
                change = already_suggested * self.feedback_change
            elif already_suggested <= self.b:  # from a to b I keep increasing the feedback as increase at a
                change = self.a * self.feedback_change
            elif already_suggested <= self.c:  # from b to c I start slowing down the increase
                # change = (already_suggested * self.feedback_change) - (self.a * self.feedback_change)
                change = self.a * self.feedback_change - ((self.a * self.feedback_change) / (self.c - self.b)) * (
                            already_suggested - self.b)
            else:  # after c the change is negative
                change = (already_suggested - self.c) * self.feedback_change * -1

            "If mood is involved"
            if (self.mood_swing_freq > 0) and self.curr_interaction % self.mood_swing_freq == 0:
                self.mood = self.mood * -1
                change = change * self.mood

            feedback_suggestion = feedback_suggestion + change

            if feedback_suggestion < 0:
                feedback_suggestion = 0
            if feedback_suggestion > 10:
                feedback_suggestion = 10

            self.feedback_dict[index_str] = feedback_suggestion

            """ Memory management """
            while (len(self.memory) > 0) and len(self.memory) >= self.mem_size:
                self.memory.pop(0)
            self.memory.append(str(activity))

            return feedback_suggestion
        return fis_feedback

    def getFeedbackForActivitiesinContexts(self, contexts, list_activities_ids, info_activities, default_feedback=-1):
        feedback = []
        for i in range(len(contexts)):
            context = self.env.getRandomEnvVal()
            c_id = 0
            for c in context:
                context[c] = contexts[i, c_id]
                c_id = c_id + 1
            prop_activity = info_activities[str(int(list_activities_ids[i]))]
            # print(list_activities_ids[i])
            # print(prop_activity)
            # print(self.actions_centers)
            prop_activity_modality = getActionCenterCrispVal(self.actions_centers, prop_activity)
            # print("Getting Feedback for activity " + str(prop_activity) + " in context " + str(getInputLinguisticInterpretation(
            #     self.variables_details, context, self.var_possible_val,
            #     self.variables_universe, self.dimensions_values)))

            feedback.append(
                self.getFeedbackForActivityInContext(context, self.actions_to_ti[prop_activity], prop_activity,
                                                     prop_activity_modality, default_feedback))
        return feedback
