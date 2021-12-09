from random import Random

from simulation.agents.patient.random_patient_agent import RandomPatientAgent


class RandomPatientAgentCreator(object):
    """
    A class used as a Factory of RandomPatientAgents. It creates a RandomPatientAgent with the appropriate paramters
    based on the type of patient indicated in patient_type of function __new__
    It supports patient types:
    - random-stubborn: a random patient that never changes its feedback regardless of how many times an activity is
    suggested consecutively
    - random-annoyed: a random patient that starts decreasing its feedback for an activity if
    that activity is suggested more than 1 time in the last 14 steps
    - random-moody: a random patient that every 14 steps changes its mood and,
    with positive mood it increases the feedback for an activity the more the activity is repeated,
    with negative mood it decreases the feedback the more the activity is repeated
    - random-repetitive: a random patient that increases the feedback more and more if activities are repeatedly
    suggested
    - random: a random patient whose parameters are randomly and uniformly sampled (every patient random seed depends on
    the id of the trial of the experiment). Potential random patients include also the types above described, except for
    the moody
    """

    def __new__(cls, patient_type, model, env, sim_param,
                ta_details, variables_details, pref_details, verbose):
        if patient_type == "random-stubborn":
            return RandomPatientAgent("Stubborn_" + str(sim_param["trial"]), model, env, sim_param, 0, 14, 14, 14, 0.01,
                                      0.6, 0, ta_details, variables_details, pref_details, verbose)
        if patient_type == "random-annoyed":
            return RandomPatientAgent("Annoyed_" + str(sim_param["trial"]), model, env, sim_param, 1, 1, 1, 14, 0.01,
                                      0.6, 0, ta_details, variables_details, pref_details, verbose)
        if patient_type == "random-moody":
            return RandomPatientAgent("Moody_" + str(sim_param["trial"]), model, env, sim_param, 3, 10, 14, 14, 0.01,
                                      0.6, 14, ta_details, variables_details, pref_details, verbose)
        if patient_type == "random-repetitive":
            return RandomPatientAgent("Repetitive_" + str(sim_param["trial"]), model, env, sim_param, 14, 14, 14, 14,
                                      0.01, 0.6, 0, ta_details, variables_details, pref_details, verbose)
        if patient_type == "random":
            r = Random()
            r.seed(sim_param["trial"])
            mem_min = 0
            mem_max = 14
            slope_min = 0.01
            slope_max = 0.05

            c = r.randint(mem_min, mem_max)
            b = r.randint(mem_min, c)
            a = r.randint(mem_min, b)

            slope = r.uniform(slope_min, slope_max)
            mood_freq = 0
            prob_positive = 0.6

            return RandomPatientAgent(
                "Random_" + str(sim_param["trial"]) + "_" + str(a) + "_" + str(b) + "_" + str(c) + "_" + str(
                    mem_max) + "_" + str(slope), model, env, sim_param, a, b, c, mem_max, slope,
                prob_positive, mood_freq, ta_details, variables_details, pref_details, verbose)

        else:
            print('No classifier of type: ' + patient_type + ' supported')
            return None
