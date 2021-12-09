from mesa import Agent

import utils.constants as Constants
from simulation.agents.utils.utils import getSimpleModalityLinguisticInterpretation


class PatientAgent(Agent):
    """ A patient agent with static preferences defined in the input preferences file
    NOTE THIS CLASS MAY BE OUT OF DATE
    """

    def __init__(self, unique_id, model, preferences, actions_centers, variables_details, verbose):
        super().__init__(unique_id, model)
        self.name = unique_id
        self.proposedActivity = []
        self.happiness = 0.0
        self.lastResponse = ""
        self.preferences = preferences
        self.actions_centers = actions_centers  # i'm assuming for now they are the same as those given to NAO. this is a dataframe like <action, lval_1, lval_2, lval_3, c_lval_1, c_lval_2, c_lval_3>
        self.verbose = verbose

    def step(self):
        c = self.model.schedule.agents[0].context
        if self.proposedActivity[0] == "question":
            if self.verbose == Constants.VERBOSE_BASIC:
                print("ok sure! let me tell you something...")
            self.happiness = 5
            self.lastResponse = "yes"
        else:
            prop_activity = self.proposedActivity[0]
            prop_activity_modality = self.proposedActivity[1]

            prop_activity_interpretation = getSimpleModalityLinguisticInterpretation(self.actions_centers, prop_activity, prop_activity_modality)

            self.happiness = float((self.preferences.loc[(self.preferences['sar_action'] == prop_activity) & (
                        self.preferences['lval'] == prop_activity_interpretation), self.unique_id]).iloc[0])

            if self.happiness >= 5:
                if self.verbose == Constants.VERBOSE_BASIC:
                    print("ok sure! let's " + str(prop_activity_interpretation) + " " + str(prop_activity))
                self.lastResponse = "yes"
            else:
                if self.verbose == Constants.VERBOSE_BASIC:
                    print("I don't really feel like " + str(prop_activity_interpretation) + " " + str(
                        prop_activity) + "...")
                self.lastResponse = "no"

        if self.verbose == Constants.VERBOSE_BASIC:
            print("HAPPINESS: " + str(self.happiness))


    def prepareForNewInteraction(self):
        self.proposedActivity = []
        self.happiness = 0.0
        self.lastResponse = ""
