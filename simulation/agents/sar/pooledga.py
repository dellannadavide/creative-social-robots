import pygad
import numpy


class PooledGA(pygad.GA):
    """ Subclass extending the pygad.GA superclass"""
    def init_outer(self, outer):
        self.outer = outer
        self.populations = []
        self.populations_fitness = []

    def cal_pop_fitness(self):
        global pool
        pop_fitness = pool.map(self.outer.fitness_wrapper, self.population)
        pop_fitness = numpy.array(pop_fitness)
        self.populations.append(self.population)
        self.populations_fitness.append(pop_fitness)
        return pop_fitness