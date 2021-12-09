""" Module containing utility functions used for the simulations """
import os
import sys


def giniSimpson(species, species_individuals, N):
    """ Function that computes the GiniSimpson Index given
    @species: a set of species
    @species_individuals: a dictionary associating individuals to species from @species
    @N: the size of the population of individuals
    """
    simpson = 0.0
    denom = (N * (N - 1))
    if denom > 0:
        for s in species:
            try:
                nsp = species_individuals[s]
                simpson = simpson + ((nsp * (nsp - 1)) / denom)
            except:
                pass
    return 1 - simpson  # the ginisimpson

def computeRuleBaseDiverstity(actions_to_ti, ta, rule_base):
    """ Function that computes the two diversity indeces of a a creative rule base @rule_base.
    One index is the ginisimpson index of the rules w.r.t. (i.e. condisdering as species) the therapies to which the activities output of the rules belong
    And the second index is the ginisimpson index of the rules w.r.t. the activities output of the rules """
    individuals_of_species = {}
    for rule in rule_base:
        action = rule.consequent[0].term.parent.label
        species_rule = actions_to_ti[action]
        if species_rule in individuals_of_species:
            individuals_of_species[species_rule] = individuals_of_species[species_rule] + 1
        else:
            individuals_of_species[species_rule] = 1

    ginisimpson_therapies = giniSimpson(ta, individuals_of_species, len(rule_base))
    individuals_of_species_therapies = individuals_of_species.copy()

    aspecies = []
    individuals_of_species = {}
    for rule in rule_base:
        species_rule = rule.consequent[0].term.parent.label  # note here the species is the action itself
        if species_rule in individuals_of_species:
            individuals_of_species[species_rule] = individuals_of_species[species_rule] + 1
        else:
            individuals_of_species[species_rule] = 1
    aspecies = individuals_of_species.keys()
    ginisimpson_activities = giniSimpson(aspecies, individuals_of_species, len(rule_base))
    individuals_of_species_activitiess = individuals_of_species.copy()

    return ginisimpson_therapies, individuals_of_species_therapies, ginisimpson_activities, individuals_of_species_activitiess


def analyze_memory(local_vars):
    """ Utility function used to analyse the memory usage"""
    os.system('cls' if os.name == 'nt' else 'clear')
    # local_vars = list(locals().items())
    for var, obj in local_vars:
        print(var, sys.getsizeof(obj))
