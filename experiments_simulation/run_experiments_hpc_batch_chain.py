"""
Module to use to run the experiments on a HPC cluster based on PBS.
Based on the defined experiments and parameters, it creates a number of batches of experiments.
Each batch is sent to the HPC as a job.
Each job will run concurrently all experiments in the batch.
To run one experiment the file run_experiment_commandline.py is invoked by the node in the HPC.

The file is called "..._chain.py" because it is possible to set a chain of batches of experiments
where given the total number N jobs (batches of experiments) to run, x of them are run at a time, and
if N>x then the script will first send x batches,
then start a timer and regularly checks if the current number of running jobs is < x. In that case it sends new jobs
to fill the available slots, until all experiments are run.
"""

import itertools as it
import subprocess
from pathlib import Path

from simulation.run_simulation import *


def submitJob(index_list, allNames, results_folder, all_experiments_timestamp, verbose, c_list):
    """
    Encoding the combination into a string of arguments
    """
    command_concat_str = ""
    exp_ind=0
    for c in c_list:
        exp_param_args = "exp_id=" + str(index_list[exp_ind]) + " "
        for i in range(len(allNames)):
            exp_param_args = exp_param_args + str(allNames[i]) + "=" + str(c[i]) + " "
        exp_param_args = exp_param_args + "results_folder=" + str(results_folder) + " all_experiments_timestamp=" + str(
            all_experiments_timestamp) + " verbose=" + str(verbose)
        # print(exp_param_args)
        command_concat_str = command_concat_str + ("" if (command_concat_str == "") else " &" ) + "python3 ./run_experiment_commandline.py "+str(exp_param_args)
        exp_ind = exp_ind + 1

    p = subprocess.Popen('qsub', stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=False)
    # Customize your options here
    job_name = "gafc_%s_%d-%d" % (all_experiments_timestamp[-3:], index_list[0], index_list[-1])
    email = "d.dellanna@tudelft.nl"
    # walltime = "00:01:00"
    processors = "nodes=1:ppn=%d" % len(index_list)
    # processors = "nodes=1:ppn=20"
    command = command_concat_str + " & wait"

    job_string = """#!/bin/bash
       #PBS -N %s
       #PBS -m abe
       #PBS -M %s
       #PBS -l %s
       #PBS -o %s${PBS_JOBNAME}.o${PBS_JOBID}
       #PBS -e %s${PBS_JOBNAME}.e${PBS_JOBID}
       cd $PBS_O_WORKDIR
       trap 'echo "%s" >> %saborted.txt' TERM
       %s""" % (job_name, email, processors, results_folder,  results_folder, str(index_list[0])+"-"+str(index_list[-1]), results_folder, command)

    # Send job_string to qsub
    out, err = p.communicate(job_string.encode())
    # print(out)
    # print(str(index_list[0])+"-"+str(index_list[-1]))
    return out

def getNumberOfJobsRunning():
    qstat = subprocess.Popen(['qselect', '-u', 'ddellanna'], stdout=subprocess.PIPE, stdin=subprocess.PIPE,
                             shell=False)
    wc = subprocess.Popen(['wc', '-l'], stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=False)
    qstat_out, qstat_err = qstat.communicate()
    wc_out, wc_err = wc.communicate(qstat_out)
    return int(wc_out)

def main():
    """
    Step-based experiments.
    Define in the following the possible values of all variables to test
    """

    now = datetime.now()
    all_experiments_timestamp = str(now.strftime("%Y%m%d%H%M%S"))
    all_experiments_identifier = "exp"
    data_folder = "../simulation/data/"
    results_folder = "../results/exp_"+all_experiments_identifier+"_" + all_experiments_timestamp + "/"
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    verbose = Constants.VERBOSE_FALSE

    """ 
    Each of the following parameter values is a list, which can contain more than one value.
    If multiple values are expressed for different parameters, then all combinations of all expressed values are tried
    """
    exp_possible_val = {
        'nr_interactions': [4000],
        'nr_steps': [1],
        'sampling_trials': [1],
        'num_generations': [50],
        'nr_rules': [20],
        'rate_parents_mating': [0.8],
        'parent_selection_type': ['sss'],
        'keep_parents': [1],
        'crossover_type': ['uniform'],
        'mutation_type': ['random'],
        'mutation_probability': [0.25],
        'crossover_probability': [1],
        'complexity_cost': [0], # todo currently not supported -> ignore
        'creativity_gain': [0], # todo currently not supported -> ignore
        'learning_rate_weights': [0.01],
        'learning_rate_credits': [0.04],  # should be >0, with 0 the system will not learn
        'default_credit': [1],
        'universes_granularity': [10],
        'compute_disabled_counterparts_credits': [False], # todo currently not supported -> ignore
        'consider_repetition_cost': [True],
        'multiplier_last_n_step_negative_repetition': [7],  # 2,7
        'last_n_steps_repetition_halving_cost': [10],  # 0,2,10
        'update_arules': [False], # todo currently not supported -> ignore
        'reassess_weights': [False], # todo currently not supported -> ignore
        'population_diversity_importance': [0.5], # should be in [0,1]
        'therapies_activities_diversity_importance_ratio': [0.75], # should be in [0,1]
        'patient_type': ['random'],
        'patient_name': ['PatientA'],
        'patient_models_file': ['none'], # to be ignored for the synthetic patients and noise experiments
        'therapeutic_actions_details_file': [data_folder + "therapeutic_actions_PATIENTNAME.csv"], # leave the _PATIENTNAME suffix, it will be replaced later
        'variables_details_file': [data_folder + "variables_simple.csv"],
        'preferences_details_file': [data_folder + "preferences_PATIENTNAME.xlsx"], # relevant only for experiments not currently reported in the paper
        'personalize': [True],
        'noise_probability': [0], # should be in [0,1]
        'noise_type': ['none'], # can be 'gaussian', 'inv_gaussian', 'reversed_feedback',
        'trial': list(range(200)) # can also be just one value [0]
    }

    """
    Here all the parameters expressed above are combined.
    Each of the possible combination is an experiment.
    Note that the parameter "trial" is included here. 
    By combining trial with the other parameters we obtain the repeated trials
    """
    exp_variating_param = []
    for k in exp_possible_val:
        if len(exp_possible_val[k]) > 1:
            exp_variating_param.append(k)
    allNames = sorted(exp_possible_val)
    combinations = it.product(*(exp_possible_val[Name] for Name in allNames))
    number_experiments = len(list(it.product(*(exp_possible_val[Name] for Name in allNames))))

    """ Parameters for spliiting the jobs in batch based on the HPC capabilities and constraints.
    max_nr_experiments_per_batch determines the maximum number of experiments to put together in a batch (to be run concurrently)
    max_nr_batches, instead, determines how many batches/jobs should be sent to the HPC at a time.
    If more batches than max_nr_batches are required in total to run all experiments, 
    then the script will first send max_nr_batches first, then start a timer and every 5 minutes checks if some
    slots are available (i.e., if the current number of running jobs is < max_nr_batches) 
    and sends new jobs to fill the available slots, until all experiments are run. """
    max_number_experiments_per_batch = 20
    max_nr_batches = 10

    """ By tweaking the following two parameters it is possible to skip some of the experiments and repeat others
    (e.g., in case some of the jobs were killed and some particular exp need to be repeated).
    Example 1:
        out of 200 experiments, those with id 5, 6, 44 need to be repeated, while all the others were ok
        set
        index_first_exp = 500 (a value higher than 200, so that all the exp will be skipped expect for those to repeat)
        exp_to_repeat = [5,6,44]
    Example 2:
        out of 200 experiments, those with id 5, 6, 44 need to be repeated, as well as all starting from id 100
        set
        index_first_exp = 100 (the index of the first exp to execute)
        exp_to_repeat = [5,6,44]
    """
    index_first_exp = 0
    exp_to_repeat = []

    """ Based on the experiments, I create all batches.
     I obtain, 
     in var batches all batches to send, and
     in var batches_exp_indeces the indeces of the exp of all the experiments in each batch
     """
    batches = []
    batches_exp_indeces = []
    exp_index = 0
    batch = []
    exp_indeces = []
    for c in combinations:
        if (exp_index >= index_first_exp) or (exp_index in exp_to_repeat):
            if len(batch)<max_number_experiments_per_batch: #if I can insert the current combination c in the current batch
                batch.append(c)
                exp_indeces.append(exp_index)
            else:  # otherwise I append the batch to the list of batches
                batches.append(batch)
                batches_exp_indeces.append(exp_indeces)
                # and I start creating the new one
                batch = [c]
                exp_indeces = [exp_index]
        exp_index = exp_index + 1
    # in case the last batch was not full and I didn't add it to the list of batches
    if len(batch) > 0:
        batches.append(batch)
        batches_exp_indeces.append(exp_indeces)
        batch = []
        exp_indeces = []

    """ Running """
    print("Running " + str(number_experiments) + " experiments in batches of max "+str(max_number_experiments_per_batch)+" [skipping the first "+str(index_first_exp-1)+", except for "+str(exp_to_repeat)+"]...")
    next_batch_to_submit = 0
    while next_batch_to_submit<len(batches):
        if getNumberOfJobsRunning()<max_nr_batches:
            jobid = submitJob(batches_exp_indeces[next_batch_to_submit], allNames, results_folder,
                              all_experiments_timestamp, verbose, batches[next_batch_to_submit])
            next_batch_to_submit = next_batch_to_submit + 1
        else:
            free_space = False
            while not free_space:
                time.sleep(60*5) # wait 5 minutes
                if int(getNumberOfJobsRunning())<max_nr_batches:
                    free_space = True

if __name__ == '__main__':
    main()
