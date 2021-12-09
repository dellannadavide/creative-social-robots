import math
import time
import traceback

import numpy
from skfuzzy import control as ctrl

from simulation.agents.sar.unpooledga import UnpooledGA
from simulation.agents.utils.fuzzycontroller import FuzzyController
from simulation.agents.utils.utils import sample_inputs, getRepetitionCost
from simulation.utils.utils import computeRuleBaseDiverstity

from statistics import *

from matplotlib import pyplot as plt

import utils.constants as Constants


class GAFC:
    """
    Class containing the necessary functions that interface a fuzzy controller with genetic algorithms.
    The genetic algorithm is used to evolve the rule base of the fuzzy controller.
    It requires to know how many values for the input/output variables of the controller
    """

    def __init__(self, param, nao):
        self.param = param
        self.nao = nao
        self.verbose = self.nao.verbose
        self.nr_val_input_var = 3
        self.nr_val_action_var = 3
        self.nr_val_eval_var = 5

    def run(self, init_population):
        """ First I resize the population in case it is bigger than expected"""
        resized_init_population = init_population
        pop = {}
        if (not (init_population is None)) and (len(init_population) > self.param["sol_per_pop"]):
            if self.verbose == Constants.VERBOSE_BASIC:
                print("the population is too big, resizing it...")
            for c in init_population:
                c_str = str(c).replace(",", "").replace(" ", "").replace("[", "").replace("]", "")
                rule = self.chromosome2rule(c)
                sol_str = str(rule)
                pop[c_str] = self.param["default_credit"]
                try:
                    pop[c_str] = self.nao.rules_credits[sol_str]
                except:
                    pass
            resized_init_population = []
            i = 0
            for c in sorted(pop.items(), key=lambda item: item[1], reverse=True):
                if i < self.param["sol_per_pop"]:
                    # print(c[0])
                    chrom_list = []
                    for bit in c[0]:
                        chrom_list.append(int(bit))
                    resized_init_population.append(chrom_list)
                    i = i + 1
                else:
                    break
            resized_init_population = numpy.array(resized_init_population)

        """ Then I instantiate the GA"""
        # Create the instance
        ga_instance = UnpooledGA(num_generations=self.param["num_generations"],
                                 # ga_instance = self.PooledGA(num_generations=self.num_generations,
                                 num_parents_mating=self.param["num_parents_mating"],
                                 fitness_func=lambda sol, sol_id: self.fitness_func(sol, sol_id),
                                 sol_per_pop=self.param["sol_per_pop"],
                                 num_genes=(len(self.nao.creative_controller_possible_inputs) * math.ceil(
                                     math.log(self.nr_val_input_var + 1, 2))) + math.ceil(
                                     math.log(4 * len(self.nao.creative_controller_possible_outputs), 2)),
                                 #  init_range_low=init_range_low,
                                 #  init_range_high=init_range_high,
                                 # gene_space = self.gene_space,
                                 init_range_low=0,
                                 init_range_high=2,
                                 random_mutation_min_val=0,
                                 random_mutation_max_val=2,
                                 mutation_by_replacement=True,
                                 gene_type=int,
                                 parent_selection_type=self.param["parent_selection_type"],
                                 keep_parents=self.param["keep_parents"],
                                 crossover_type=self.param["crossover_type"],
                                 mutation_type=self.param["mutation_type"],
                                 #  mutation_percent_genes=mutation_percent_genes,
                                 # on_generation=on_gen,
                                 mutation_probability=self.param["mutation_probability"],
                                 crossover_probability=self.param["crossover_probability"],  # )
                                 initial_population=resized_init_population)
        # Instantiate the super class of the instance of GA
        ga_instance.init_outer(self, self.nao.verbose)
        """ Then I run the GA"""
        global pool
        # print("number of cores "+str(mp.cpu_count()))
        start_time = time.time()
        ga_instance.run()
        if self.verbose == Constants.VERBOSE_BASIC:
            print("AFTER GA TIME ELAPSED --- %s seconds ---" % (time.time() - start_time))

        """ Finally I determine the best population, and clean it to remove duplicates and invalid rules.
        The best population is the population
        bestpop =  argmax_{p in populations} (popfit*(1-divimp) + popdiv*divimp)
        where
        popfit~ is the average fitness of the rules in a population $p$, normalized to the range $[0,1]$ to be 
        commensurable with popdiv, 
        popdiv~ is a diversity index of the population $p$, s.t.
        popdiv = g_{p,ther}*(1-actTherImp) + g_{p,act}*actTherImp
        where $actTherImp in [0,1]$ is a parameter that determines the relative importance of the two different factors. 
        and g_{p,ther} and g_{p,act} are the gini-indeces characterizing the diversity w.r.t. therapies and activities
        and $divimp in [0,1]$ is the relative importance given to the diversity index w.r.t. the fitness (gamma, below).
        """
        unique_populations = []
        unique_populations_fitness = []
        best_population_val = -1000000000
        best_population_index = -1
        """ for every population tried in the evolutionary process"""
        for i in range(len(ga_instance.populations)):
            up, uf = self.getFitnessOfUniqueValidIndividuals(ga_instance.populations[i],
                                                             ga_instance.populations_fitness[i])
            # print(uf)
            unique_populations.append(up)
            unique_populations_fitness.append(uf)
            """ convert the population into a rule base"""
            rulebase = self.population2rulebase(up)
            """ compute the diversities"""
            diversity_index_therapies, individuals_per_species_therapies, \
            diversity_index_activities, individuals_per_species_activities = \
                computeRuleBaseDiverstity(self.nao.actions_to_ti, self.nao.ta, rulebase)
            """ calculate the value of the i-th population"""
            gamma = self.param["population_diversity_importance"]
            avg_fitness = numpy.mean(uf)
            max_eval = 10
            min_eval = 0
            for v in self.nao.eval_var:
                max_eval = self.nao.variables_universe[v][-1]
                min_eval = self.nao.variables_universe[v][0]
                break  # todo assuming that every eval var is the same
            norm_avg_fitness = (avg_fitness - min_eval) / (max_eval - min_eval)
            kappa = self.param["therapies_activities_diversity_importance_ratio"]
            diversity = diversity_index_therapies * (1 - kappa) + diversity_index_activities * kappa
            # pop_val = (avg_fitness*(1-gamma)) + (diversity*gamma)
            pop_val = (norm_avg_fitness * (1 - gamma)) + (diversity * gamma)

            if pop_val > best_population_val:
                best_population_val = pop_val
                best_population_index = i

        if self.verbose == Constants.VERBOSE_BASIC:
            print("Fitness value of the best population = {solution_fitness}".format(
                solution_fitness=best_population_val))
            print("Generation of the best population : {solution_idx}".format(solution_idx=best_population_index))

        """Create and return the new rule base form the best population found"""
        best_unique_population = unique_populations[best_population_index]
        creative_rulebase = self.population2rulebase(best_unique_population)
        return creative_rulebase, ga_instance.populations[best_population_index]

    def rule2chromosome(self, rule_info):
        """
        Function that given the info about the variables in the input/output of a rule,
        returns the chromosome that encodes the rule.

        @rule_info: is a list like [p, c, e] where
        p is a list of pairs [var, val] describing all terms in the premise of the rule
        c is a list of pairs [var, val] describing all terms in the consequent of the rule (only one pair expected)

        @chromosome is a binary string chromosome encoding a c-rule. It is made of
        $(\sum_{i\in \cinvar}\lceil log_2(v_i+1)\rceil) + \lceil log_2(\sum_{a\in \act}v_a)\rceil $ bits (genes).

        The first $\sum_{i\in \cinvar}\lceil log_2(v_i+1)\rceil$ genes characterize the premise of the rule,
        which can be composed by at most $|\cinvar|$ terms (corresponding to the number of possible input variables).
        The $i$-th term of the premise is represented by means of $\lceil log_2(v_i+1)\rceil$ genes.

        The remaining $\lceil log_2(\sum_{a\in \act}v_a)\rceil$ genes characterize the consequent of the rule.
        Since we consider rules with only one output (activity), we encode the \textit{index} of a linguistic value
        from the ordered list \coutvar~ of all possible realizations of all output variables (activities) for the
        c-rules, so that, given the order of \coutvar, the encoded index indicates both the linguistic variable
        and its membership function.
        """

        """ Premise """
        default_input_val = ""
        for i in range(math.ceil(math.log(self.nr_val_input_var + 1, 2))):
            default_input_val = default_input_val + "0"
        chromosome_premise = [default_input_val] * len(self.nao.creative_controller_possible_inputs)
        for t in rule_info[0]:  # for every term in the premise
            term_var = t[0]
            term_val = t[1]
            for i in range(len(self.nao.creative_controller_possible_inputs)):
                var = self.nao.creative_controller_possible_inputs[i]
                if var == term_var:
                    for j in range(len(self.nao.dimensions_values[var])):
                        val = self.nao.dimensions_values[var][j]
                        if val == term_val:
                            chromosome_premise[i] = format(j + 1,
                                                           "0" + str(math.ceil(math.log(self.nr_val_input_var + 1,
                                                                                        2))) + "b")  # +1 because the 00 is reserved for the disabled value. 02b means: 2 bits, where the initial 0s expressed
                            break
        """ Consequent """
        nr_genes_output = math.ceil(math.log(len(self.nao.concat_output_membership_functions), 2))
        for t in rule_info[1]:  # for every term in the consequent
            term_var = t[0]
            term_val = t[1]
            ov = ""
            for i in range(nr_genes_output):
                ov += "0"
            i = 0
            changed = False
            for o in self.nao.creative_controller_possible_outputs:
                if o == term_var:
                    j = 0
                    for v in self.nao.dimensions_values[o]:
                        if v == term_val:
                            ov = format(i + j + 1, "0" + str(nr_genes_output) + "b")
                            changed = True
                            break
                        j = j + 1
                    break
                else:
                    i = i + len(self.nao.dimensions_values[o])
                if changed:
                    break

        """ Construct and return the chromosome """
        chromosome_premise_str = ""
        for e in chromosome_premise:
            chromosome_premise_str += e
        chromosome = chromosome_premise_str + ov
        return chromosome

    def rulebase2population(self, info_rulebase):
        """ Function that transforms a rule base into a population of chromsomomes """
        population = None
        for ir in info_rulebase:
            if population is None:
                population = []
            population.append(self.rule2chromosome(ir))
        if not population is None:
            pop_list = []
            for c in population:
                chrom_list = []
                for bit in c:
                    chrom_list.append(int(bit))
                pop_list.append(chrom_list)
            return numpy.array(pop_list)
        return None

    def chromosome2rule(self, chromosome):
        """ Function that transforms a chromosome into a rule.
        @chromosome is a binary string chromosome encoding a c-rule. It is made of
        $(\sum_{i\in \cinvar}\lceil log_2(v_i+1)\rceil) + \lceil log_2(\sum_{a\in \act}v_a)\rceil $ bits (genes).

        The first $\sum_{i\in \cinvar}\lceil log_2(v_i+1)\rceil$ genes characterize the premise of the rule,
        which can be composed by at most $|\cinvar|$ terms (corresponding to the number of possible input variables).
        The $i$-th term of the premise is represented by means of $\lceil log_2(v_i+1)\rceil$ genes.

        The remaining $\lceil log_2(\sum_{a\in \act}v_a)\rceil$ genes characterize the consequent of the rule.
        Since we consider rules with only one output (activity), we encode the \textit{index} of a linguistic value
        from the ordered list \coutvar~ of all possible realizations of all output variables (activities) for the
        c-rules, so that, given the order of \coutvar, the encoded index indicates both the linguistic variable
        and its membership function.
        """
        """ Premise """
        antecedent = {}
        ante = None
        for i in range(0, len(self.nao.creative_controller_possible_inputs)):
            starting_index_i = math.ceil(math.log(self.nr_val_input_var + 1, 2)) * i
            binary_ith_in = ""
            for bit_ix in range(math.ceil(math.log(self.nr_val_input_var + 1, 2))):
                binary_ith_in = binary_ith_in + str(chromosome[bit_ix + starting_index_i])
            ith_input_mf_index = int(binary_ith_in, 2) - 1
            if ith_input_mf_index == -1:  # ignore the i-th input for this rule
                continue
            ling_var_name = self.nao.creative_controller_possible_inputs[i]
            ling_var = self.nao.creative_controller_inputs_mf[ling_var_name]
            ling_val = self.nao.dimensions_values[ling_var_name][ith_input_mf_index]
            term = ling_var[ling_val]
            if ante is None:
                ante = term
            else:
                ante = ante & term
            antecedent[ling_var_name] = ling_val

        """ Consequent """
        binary_out = chromosome[(math.ceil(math.log(self.nr_val_input_var + 1, 2)) * len(
            self.nao.creative_controller_possible_inputs)):len(chromosome)]
        binary_out_str = ""
        for bit in binary_out:
            binary_out_str = binary_out_str + str(bit)
        output_mf_index = int(binary_out_str, 2)
        if output_mf_index >= len(self.nao.concat_output_membership_functions):
            return None, antecedent, [], True
        output_ling_val = self.nao.concat_output_membership_functions[output_mf_index]
        output_ling_var = None
        output_ling_var_name = ""
        i = 0
        for a in self.nao.creative_controller_possible_outputs:
            i = i + len(self.nao.dimensions_values[a])
            if output_mf_index < i:
                # I found the action
                output_ling_var = self.nao.creative_controller_outputs_mf[a]
                output_ling_var_name = a
                break  # I can break I don't need to continue
        conseq = output_ling_var[output_ling_val]
        consequent = [output_ling_var_name, output_ling_val]

        if (ante is None):  # or output_ling_val == 'disabled':
            return None, antecedent, consequent, True
        return ctrl.Rule(ante, conseq), antecedent, consequent, False

    def population2rulebase(self, population):
        """ Function that transforms a population of chromsomomes into a rule base """
        rulebase = []
        for c in population:
            creative_rule, antecedent_info, consequent_info, discard = self.chromosome2rule(c)
            if not discard:
                rulebase.append(creative_rule)
        return rulebase

    def getFitnessOfUniqueValidIndividuals(self, pop, pop_fit):
        """ Function that returns the fitness of the valid individuals in a population where duplicates are removed """
        si = []
        up = []
        uf = []
        for i in range(len(pop)):
            if pop_fit[i] > -100:
                if not str(pop[i]) in si:
                    up.append(pop[i])
                    uf.append(pop_fit[i])
                    si.append(str(pop[i]))
        return up, uf

    def fitness_func(self, solution, solution_idx):
        """ Function that computes the fitness of a chromosome
        @solution is the chromosome

        The fitness is computed as per
        \fit = \credit\cdot \wAct\cdot \rass\cdot \frac{1}{\repcost^{\rep}}
        where \credit~ is the credit of $r$ at time $t$
        (if the rule was never considered before, by default $c_{r,t}=1$),
        \wAct~ is the weight of the therapy to which the activity $a$, output of $r$, belongs,
        \rep~ is the number of times the activity $a$ has been already suggested to the patient in the last $n$ suggestions,
        \repcost~ is a parameter used to determine the cost to associate to multiple repetitions of the same suggestion,
        and finally \rass~ is an assessment of rule $r$ obtained by means of the assessor A-FIS via a monte carlo approach.
        """

        sampling_outputs = []
        """ First decode the chromosome into a rule, and possibly immediately return -M if to discard"""
        creative_rule, antecedent_info, consequent_info, discard = self.chromosome2rule(solution)
        if discard:
            return self.fitness(creative_rule, None, antecedent_info, consequent_info, discard)

        """ Then create a C-FIS with only one rule (the decoded rule) """
        creative_controller = FuzzyController(self.nao.dimensions_values,
                                              self.nao.variables_universe,
                                              self.nao.fuzzysets_values,
                                              self.nao.variables_default_val,
                                              self.nao.creative_controller_possible_inputs,
                                              self.nao.creative_controller_possible_outputs,
                                              [creative_rule])
        """ Then adopt Monte Carlo approach to estimate, via the assessor controller, an assessment of the rule """
        try:
            for i in range(self.param["sampling_trials"]):
                sampled_inputs = sample_inputs(True, i, self.nao.curr_interaction, self.nao.variables_default_val,
                                               self.nao.action_var, self.nao.fuzzysets_values,
                                               self.nao.variables_universe, rule_antecedent_info=antecedent_info)
                assout = self.assess_creative_controller(sampled_inputs, creative_controller)
                if len(assout) == 0:
                    ta = self.nao.actions_to_ti[consequent_info[0]]
                    for v in self.nao.assessors_possible_outputs[ta]:
                        assout.append(self.nao.variables_default_val[v])
                output_eval = mean(assout)
                sampling_outputs.append(output_eval)
                if self.verbose > Constants.VERBOSE_BASIC:
                    for assessor in self.nao.assessors:
                        print(self.nao.assessors[assessor].controlsystem.input)
                        # relaxed.view(sim=assessor)
                        print(self.nao.assessors[assessor].controlsystem.output)
                    print(assout)
                    print(output_eval)
        except:  # I do it only once because, since we had an error, we will have it every time
            traceback.print_exc()
            assout = []
            ta = self.nao.actions_to_ti[consequent_info[0]]
            for v in self.nao.assessors_possible_outputs[ta]:
                assout.append(self.nao.variables_default_val[v])
            sampling_outputs.append(mean(assout))

        """
        Here actually compute, and then return, the fitness
        """
        fit = self.fitness(creative_rule, mean(sampling_outputs), antecedent_info, consequent_info, False)
        return fit

    def assess_creative_controller(self, inputs, creative_controller):
        """ Function that returns an assessment of a creative controller @creative_controller
         obtained with the asessor controller with inputs @inputs and the output of the @creative_controller when
         given inputs @inputs """
        c_out, rules_activations = creative_controller.computeOutput(inputs, False)
        if self.verbose > Constants.VERBOSE_BASIC:
            print("assessing the creative controller")
            print("inputs")
            print(inputs)
            print("creative inputs")
            print(creative_controller.controlsystem._get_inputs())
            print("creative outputs")
            print(c_out)

        output_assessors = {}
        a_inputs = dict(inputs)
        for cout in c_out:
            a_inputs[cout] = c_out[cout]
            ta = self.nao.actions_to_ti[cout]
            a_out, a_rules_activations = self.nao.assessors[ta].computeOutput(a_inputs)
            for ao in a_out:
                if ao in output_assessors:
                    output_assessors[ao].append(a_out[ao])
                else:
                    output_assessors[ao] = [a_out[ao]]

        assout = [mean(output_assessors[ao]) for ao in output_assessors]
        return assout

    def fitness(self, rule, ass, antecedent_info, consequent_info, to_discard):
        """ The actual function that computes the fitness of a rule.
        The fitness is computed as per
        \fit = \credit\cdot \wAct\cdot \rass\cdot \frac{1}{\repcost^{\rep}}
        where \credit~ is the credit of $r$ at time $t$
        (if the rule was never considered before, by default $c_{r,t}=1$),
        \wAct~ is the weight of the therapy to which the activity $a$, output of $r$, belongs,
        \rep~ is the number of times the activity $a$ has been already suggested to the patient in the last $n$ suggestions,
        \repcost~ is a parameter used to determine the cost to associate to multiple repetitions of the same suggestion,
        and finally \rass~ is an assessment of rule $r$ obtained by means of the assessor A-FIS via a monte carlo approach.
        """
        fitness = -100
        if to_discard:
            return fitness
        sol_str = str(rule)
        rule_credit = self.param["default_credit"]
        try:
            rule_credit = self.nao.rules_credits[sol_str]
        except:
            pass
        action = consequent_info[0]
        if self.verbose == Constants.VERBOSE_VERY_HIGH:
            if rule_credit > 1:
                print("credit rule with " + str(action) + ": " + str(rule_credit))
        w_ta = self.nao.weights_therapeutic_interventions[self.nao.actions_to_ti[action]]
        repetition_cost = getRepetitionCost(action, self.nao.logInteractions, self.param)
        fitness = (rule_credit * w_ta * ass) / repetition_cost
        if self.verbose == Constants.VERBOSE_VERY_HIGH:
            if repetition_cost > 1:
                print("fitness of rule with " + str(action) + " WITHOUT repetition cost: " + str(
                    rule_credit * w_ta * ass))
                print(repetition_cost)
                print("fitness of rule with " + str(action) + "WITH repetition cost: " + str(fitness))
        return fitness


    def plot_result(self, title="PyGAD - Generation vs. Population Fitness", xlabel="Generation",
                    ylabel="Population Fitness", linewidth=3, save_dir=None, unique_populations_fitness=None):
        """ Utility function to create a plot with the results of the GA execution"""
        fig = plt.figure()
        x = range(len(unique_populations_fitness))
        pop_fitness_averages = []
        pop_fitness_errors = []
        for p in unique_populations_fitness:
            pop_fitness_averages.append(numpy.mean(p))
            pop_fitness_errors.append(numpy.std(p))
        pop_fitness_averages = numpy.array(pop_fitness_averages)
        pop_fitness_errors = numpy.array(pop_fitness_errors)
        plt.errorbar(x, pop_fitness_averages, pop_fitness_errors, linewidth=linewidth, alpha=0.5)
        plt.plot(x, pop_fitness_averages, linewidth=linewidth)
        plt.fill_between(x, pop_fitness_averages - pop_fitness_errors, pop_fitness_averages + pop_fitness_errors,
                         alpha=0.2)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
        return fig
