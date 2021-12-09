from statistics import mean

# from matplotlib.animation import FuncAnimation
from mesa import Agent
from random import Random

import utils.constants as Constants
from simulation.agents.utils.fuzzycontroller import FuzzyController
from simulation.agents.utils.utils import importDataFromFiles, sample_inputs, getSimpleModalityLinguisticInterpretation, \
    getSimpleInputLinguisticInterpretation


# import matplotlib as mpl
# import matplotlib.pyplot as plt
# mpl.use('TkAgg')  # or can use 'TkAgg', whatever you have/prefer


class RandomPatientAgent(Agent):
    """
    Class representing a random patient interacting with the SAR.
    Every Patient, when receiving a suggestion for an activity, returns a numerical value $f\in[0,10]$,
    where 0 indicates an extremely negative feedback, while 10 indicates an extremely positive one.
    Initially, $f$ is randomly selected from $[0, 10]$ according to a distribution s.t.
    in 60% of cases $f< 5$ and in the remaining 40% of cases $f>=5$.
    Whenever the same activity is suggested again by the SAR, the Patient adjusts the previous feedback according to its
    \textit{type}. We distinguish different types of Patients based on how they react to repeated suggestions of the
    same activity over time. In particular, we consider Patients that have a \textit{memory} of size $m$
    (i.e., they remember the previous $m$ suggestions of the SAR).
    Given an activity $a$, every Patient returns a feedback $f$ that is equal to the feedback that was returned the most
    recent time the activity was suggested, adjusted by $\Delta_f(\repinmem)$. We have:
    f \gets f + \Delta f(\repinmem)
    where
      \Delta f(\repinmem)=
        \delta \repinmem, & \text{if $\repinmem\leq x$}.\\
        \delta x, & \text{if $x< \repinmem\leq y$}.\\
        \delta x - \frac{\delta x}{z-y}(\repinmem-y), & \text{if $y< \repinmem\leq z$}.\\
        -\delta(\repinmem-z), & \text{if $\repinmem>y$}.

    where $\repinmem$ is the number of times that the activity $a$ was already suggested in the last $m$ steps.
    """
    def __init__(self, unique_id, model, env, sim_param, a, b, c, mem_size, feedback_change, neg_ratio, mood_swing_freq,
                 ta_details, variables_details, pref_details, verbose):
        super().__init__(unique_id, model)
        self.name = unique_id
        # print(self.name.replace("Random_", "").replace("_", "\t"))
        self.env = env
        self.proposedActivity = []
        self.feedback = 0.0
        self.lastResponse = ""
        self.actions_centers = ta_details  # todo i'm assuming for now they are the same as those given to SAR. this is a dataframe like <action, lval_1, lval_2, lval_3, c_lval_1, c_lval_2, c_lval_3>
        self.variables_details = variables_details  # a dataframe from the excel file
        self.verbose = verbose

        self.r = Random()
        self.r.seed(sim_param["trial"])

        self.a = a
        self.b = b
        self.c = c
        self.neg_ratio = neg_ratio
        self.memory = []
        self.mem_size = mem_size
        self.feedback_change = feedback_change
        self.mood_swing_freq = mood_swing_freq
        self.mood = 1
        if (self.mood_swing_freq > 0):
            if self.r.uniform(0, 1)>=0.5:
                self.mood=-1

        self.feedback_dict = {}

        self.curr_interaction = 0

        # print("..................................")
        # print("Created random patient "+str(self.name))
        # print("a = %d, b = %d, c = %d, mem_size = %d, mood = %s, slope = %s" % (self.a,self.b,self.c,self.mem_size,str(self.mood),str(self.feedback_change)))
        # print("..................................")


    #     if self.verbose <= Constants.VERBOSE_BASIC:
    #         candidates = ["macosx", "qt5agg", "gtk3agg", "tkagg", "wxagg"]
    #         for candidate in candidates:
    #             try:
    #                 plt.switch_backend(candidate)
    #                 print('Using backend: ' + candidate)
    #                 break
    #             except (ImportError, ModuleNotFoundError):
    #                 pass
    #
    #         plt.ion()
    #         self.fig = plt.figure()
    #         self.axes = self.fig.add_subplot(111)
    #         self.axes.set_ylim(0, 10)
    #         plt.title("feedback of "+str(self.name))
    #
    #         self.x, self.y = [], []
    #         # self.anim = FuncAnimation(self.fig, self.animate, interval=500)
    #
    # def animate(self):
    #     self.x.append(self.curr_interaction)
    #     self.y.append(self.feedback)
    #     plt.xlim(self.curr_interaction - 300, self.curr_interaction + 300)
    #     self.axes.plot(self.x, self.y, color="red")
    #     self.fig.canvas.draw()
    #     self.fig.canvas.flush_events()

    def step(self):
        """
        Step function executed at every simuation step
        """
        if self.proposedActivity[0] == "question":
            if self.verbose == Constants.VERBOSE_BASIC:
                print("ok sure! let me tell you something...")
            self.feedback = 5
            self.lastResponse = "yes"
        else:
            context_interpr = getSimpleInputLinguisticInterpretation(self.variables_details, self.env.context)
            context_interpr = "context" #todo note hack to force reaction regardless of the context
            if self.verbose >= Constants.VERBOSE_VERY_LOW:
                print("!!!Warning: patient hacked so to have reaction regardless of the context!!")
            prop_activity = self.proposedActivity[0]
            prop_activity_modality = self.proposedActivity[1]
            prop_activity_modality_interpretation = getSimpleModalityLinguisticInterpretation(self.actions_centers,
                                                                                              prop_activity,
                                                                                              prop_activity_modality)
            index_str = str(context_interpr) + "--" + str(prop_activity) + "-" + str(
                prop_activity_modality_interpretation)

            """ First time we are seeing the activity"""
            if not index_str in self.feedback_dict:
                rv = self.r.uniform(0, 1)
                if rv > self.neg_ratio:  # in 40% of cases I get a positive initial feedback
                    self.feedback_dict[index_str] = self.r.uniform(5, 10)  # todo assuming feedback between 0 and 10
                else:  # in 60% of cases I get a negative one
                    self.feedback_dict[index_str] = self.r.uniform(0, 5)  # todo assuming feedback between 0 and 10

            self.feedback = self.feedback_dict[index_str]

            """ After some time I adjust the feedback"""
            already_suggested = self.memory.count(index_str)
            # print("already suggested "+str(already_suggested)+" times")
            change = 0.0
            if already_suggested <= self.a:  # before reaching a repetition I keep increasing the feedback more and more
                change = already_suggested * self.feedback_change
            elif already_suggested <= self.b:  # from a to b I keep increasing the feedback as increase at a
                change = self.a * self.feedback_change
            elif already_suggested <= self.c:  # from b to c I start slowing down the increase
                # change = (already_suggested * self.feedback_change) - (self.a * self.feedback_change)
                change = self.a * self.feedback_change - ((self.a * self.feedback_change)/(self.c - self.b)) * (already_suggested-self.b)
            else:  # after c the change is negative
                change = (already_suggested - self.c) * self.feedback_change * -1

            "If mood is involved"
            if (self.mood_swing_freq > 0) and self.curr_interaction % self.mood_swing_freq == 0:
                self.mood = self.mood*-1
                change = change*self.mood

            self.feedback = self.feedback + change

            if self.feedback < 0:
                self.feedback = 0
            if self.feedback > 10:
                self.feedback = 10

            self.feedback_dict[index_str] = self.feedback

            """ Memory management """
            while (len(self.memory) > 0) and len(self.memory) >= self.mem_size:
                self.memory.pop(0)
            self.memory.append(index_str)

            """ Answering back"""
            if self.feedback >= 5:
                if self.verbose == Constants.VERBOSE_BASIC:
                    print("ok sure! let's " + str(prop_activity_modality_interpretation) + " " + str(prop_activity))
                self.lastResponse = "yes"
            else:
                if self.verbose == Constants.VERBOSE_BASIC:
                    print("I don't really feel like " + str(prop_activity_modality_interpretation) + " " + str(
                        prop_activity) + "...")
                self.lastResponse = "no"

        # if self.verbose <= Constants.VERBOSE_BASIC:
        #     print("feedback: " + str(self.feedback))
        #     self.animate()

    def prepareForNewInteraction(self):
        self.proposedActivity = []
        self.feedback = 0.0
        self.lastResponse = ""
        self.curr_interaction = self.curr_interaction + 1
