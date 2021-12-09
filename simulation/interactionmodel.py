from mesa import Model
from mesa.time import BaseScheduler
from simulation.agents.env_agent import EnvironmentAgent
from simulation.agents.patient.random_patient_agent_creator import RandomPatientAgentCreator
from simulation.agents.patient.fuzzy_patient_agent import FuzzyPatientAgent
from simulation.agents.patient.basic_patient_agent import PatientAgent
from simulation.agents.sar.sar_agent import SARAgent



class NaoInteractionModel(Model):
    """ Class Implementing a MAS (Model from the mesa library), which schedules and executes
    the environment, the patient, and the SAR agents, in this order, at every step. """
    def __init__(self, N, filename, sim_param, patient_pref, ta_details, variables_details, verbose, pref_details=None):
        if N!=1:
          raise ValueError('supported only 1 interacting agent for now. Nao will interact first')
        self.schedule = BaseScheduler(self)
        # Create agents
        env = EnvironmentAgent('Env', variables_details, self, sim_param, verbose)
        if sim_param["patient_type"] == "base":
            patient = PatientAgent(sim_param["patient_name"], self, patient_pref, ta_details, variables_details, verbose)
        if sim_param["patient_type"] == "fuzzy":
            patient = FuzzyPatientAgent(sim_param["patient_name"], self, env, sim_param, ta_details, variables_details, pref_details, verbose)
            nao = SARAgent('Nao', self, patient, env, filename, sim_param, ta_details, variables_details, verbose,
                           pref_details=pref_details)
        if "random" in sim_param["patient_type"]:
            patient = RandomPatientAgentCreator(sim_param["patient_type"], self, env, sim_param, ta_details, variables_details, pref_details, verbose)
            nao = SARAgent('Nao', self, patient, env, filename, sim_param, ta_details, variables_details, verbose)


        # NOTE: ORDER MATTERS
        self.schedule.add(env)
        self.schedule.add(nao)
        self.schedule.add(patient)


    def step(self):
        '''Advance the model by one step.'''
        self.schedule.step()