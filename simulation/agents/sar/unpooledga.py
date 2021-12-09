import pygad
import numpy


class UnpooledGA(pygad.GA):
    """ Subclass extending the pygad.GA superclass"""
    def init_outer(self, outer, verbose):
        self.outer = outer
        self.populations = []
        self.populations_fitness = []
        self.verbose = verbose

    def cal_pop_fitness(self):
        if self.valid_parameters == False:
            raise ValueError(
                "ERROR calling the cal_pop_fitness() method: \nPlease check the parameters passed while creating an instance of the GA class.\n")
        pop_fitness = []
        # Calculating the fitness value of each solution in the current population.
        for sol_idx, sol in enumerate(self.population):
            fitness = self.fitness_func(sol, sol_idx)
            pop_fitness.append(fitness)

        pop_fitness = numpy.array(pop_fitness)
        self.populations.append(self.population)
        self.populations_fitness.append(pop_fitness)
        return pop_fitness

