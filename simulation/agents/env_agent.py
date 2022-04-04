from mesa import Agent
from random import Random

import utils.constants as Constants
from simulation.agents.utils.utils import getSimpleInputLinguisticInterpretation


class EnvironmentAgent(Agent):
    """ Class representing the Environment Agent.
    At every step the environment agent randomly samples the values of the contextual variables from their
    universe of discourse
    """
    def __init__(self, unique_id, variables, model, sim_param, verbose):
        super().__init__(unique_id, model)
        self.variables = variables
        contextual_var_names = variables.loc[variables['vartype'] == "context", 'varname'].unique()
        self.context = {}
        self.contextual_val_bounds = {}
        for c in contextual_var_names:
            self.context[c] = variables.loc[(variables['varname'] == c), 'c_lval_2'].iloc[0]
            self.contextual_val_bounds[c] = [variables.loc[(variables['varname'] == c), 'c_lval_1'].iloc[0],
                                             variables.loc[(variables['varname'] == c), 'c_lval_3'].iloc[0]]
        self.verbose = verbose
        self.r = Random()
        self.curr_step = 0
        self.param = sim_param

    def step(self):
        # here we would like to advance the context. For now random
        d = 0
        if ("trial" in self.param) and ("nr_interactions" in self.param) and ("nr_steps" in self.param):
            d = self.param["trial"]*self.param["nr_interactions"]*self.param["nr_steps"]
        self.r.seed(self.curr_step+d)
        random_env = self.getRandomEnvVal()
        for c in random_env:
            self.context[c] = random_env[c]

        if self.verbose == Constants.VERBOSE_BASIC:
            print(self.context)
            print(getSimpleInputLinguisticInterpretation(self.variables, self.context))

        """ Preparing for next step """
        self.curr_step = self.curr_step + 1

    def getRandomEnvVal(self):
        random_env_val = {}
        for c in self.context:
            random_env_val[c] = self.r.uniform(self.contextual_val_bounds[c][0], self.contextual_val_bounds[c][1])
        return random_env_val
