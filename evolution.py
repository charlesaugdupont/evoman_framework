##################################################################################
# This file could contain the main methods for doing simulation, evolution, 
# the genetic algorithm, etc.
##################################################################################

import numpy as np
from numpy.lib.function_base import select
from individual import Individual

def initialize_generation(environment, population_size, num_genes):
	"""
	Randomly initializes a generation by returning list of objects of class Individual.
	:param population_size: number of individuals in population
	:param num_genes: total number of weights in neural network controller
	"""
	# initialize all individuals in the population 
	all_genotypes = np.random.uniform(-1, 1, (population_size, num_genes))
	all_sigmas = np.random.uniform(0.1, 1.0, (population_size,))
	generation = [Individual(all_genotypes[i], all_sigmas[i]) for i in range(population_size)]

	# compute fitness of all individuals
	for individual in generation:
		individual.fitness = individual.compute_fitness(environment)

	return generation


def generate_next_generation(environment, population):
	"""
	Generates next generation from current population.
	:param population: list of objects of class Individual
	"""

	# generate pairs of parents that can be used for recombination
	#parent_pairs = parent_selection_method_1(population, num_pairs=int(len(population)/2))
	parent_pairs = parent_selection_method_2(population, num_pairs=int(len(population)/2))

	# generate offspring
	offspring = []
	for i in range(len(parent_pairs)):
		children = create_offspring(environment, parent_pairs[i][0], parent_pairs[i][1], num_offspring=1)
		offspring += children # concatenate children to offspring list

	# perform survival selection to return next generation with same size as input generation
	new_population = survival_selection(population, offspring)

	return new_population

######## PARENT SELECTION MECHANISMS ########
def parent_selection_method_1(population, num_pairs):
	"""
	Returns a list of parent pairs with size num_pairs using repeated random sampling
	of 5 individuals, and selecting top 2.
	:param population: list of objects of class Individual
	:param num_pairs: how many parent pairs to return
	"""
	# iterate until we have as many parent pairs as the size of our population
	parent_pairs = []
	while len(parent_pairs) != num_pairs:
		# draw random sample of 5 individuals
		sample = np.random.choice(population, 5, replace=False)

		# select 2 parents with highest fitness scores
		top_2 = sorted(sample, key=lambda individual: individual.fitness)[-2:]
		parent_pairs.append((top_2[0], top_2[1]))

	return parent_pairs

def parent_selection_method_2(population, num_pairs):
	"""
	Returns a list of parent pairs using fitness-proportionate probabilistic sampling ("roulette wheel" method).
	:param population: list of objects of class Individual
	:param num_pairs: how many parent pairs to return
	"""
	fitness_scores = [individual.fitness for individual in population]
	minimum, maximum = min(fitness_scores), max(fitness_scores)
	# compute selection probabilities
	selection_probs = [max(0.00000001, (f-minimum)/(maximum-minimum)) for f in fitness_scores]
	# normalize such that they sum to 1.0
	selection_probs = [p/sum(selection_probs) for p in selection_probs]
	parent_pairs = []
	while len(parent_pairs) != num_pairs:
		selection = np.random.choice(population, 2, replace=False, p=selection_probs)
		parent_pairs.append((selection[0], selection[1]))

	return parent_pairs
#############################################


def create_offspring(environment, parent_1, parent_2, num_offspring):
	"""
	Generate num_offspring from parent_1 and parent_2 using recombination and mutation
	:param environment: (simulation) environment in which recombination happens
	:param parent_1: first parent object of class Individual
	:param parent_2: first parent object of class Individual
	:param num_offspring: number of offspring to generate from the parent pair
	This function applies whole arithmetic recombination (Eiben & Smith, 2015, p. 66)
	"""

	# create num_offspring children from the parent pair
	children = []
	for i in range(num_offspring):

		# apply whole arithmetic recombination to create children
		child = recombine(parent_1, parent_2)

		# apply mutation and add child to children list		
		child.mutate_self_adaptive1()
    
		# compute child's fitness after mutation
		child.fitness = child.compute_fitness(environment)
		children.append(child)

	return children


def recombine(parent_1, parent_2):
	"""
	Performs recombination between two parents, creating a child.
	Choose recombination method:
	- whole_arith_recombination(parent_1, parent_2)
	- blended_crossover
	- blended_crossover_v2
	"""
	child_genotype, child_sigma = blended_crossover(parent_1, parent_2)

	# return new child object
	return Individual(child_genotype, child_sigma)


def whole_arith_recombination(parent_1, parent_2):
	"""
	"""
	# compute child genotype
	alpha = np.random.uniform(0,1)
	child_genotype = alpha * parent_1.genotype + (1 - alpha) * parent_2.genotype

	# compute child sigma
	# Eiben & Smith don't say anything about how sigma should be handled in
	# recombination, so I chose to take the average of the two parent sigmas
	child_sigma = np.average((parent_1.sigma, parent_2.sigma))

	return child_genotype, child_sigma


def blended_crossover(parent_1, parent_2):
	"""
	ref. A Crossover Operator Using Independent Component Analysis for Real-Coded Genetic Algorithms (Takahashi & Kita, 2001)
	Can choose between two sigma methods:
	- child_sigma_v1(parent_1, parent_2)
	- child_sigma_v2(parent_1, parent_2)
	"""
	alpha = 0.5 # ref Eshelmann & Schafer
	# alpha = 0.366 # ref. (Takahashi & Kita, 2001)

	child_genotype = []
	for i in range(parent_1.num_genes):
		difference = abs(parent_1.genotype[i] - parent_2.genotype[i])
		bound_1 = min(parent_1.genotype[i], parent_2.genotype[i]) - alpha * difference
		bound_2 = max(parent_1.genotype[i], parent_2.genotype[i]) + alpha * difference
		child_genotype.append(np.random.uniform(bound_1, bound_2))

	child_sigma = child_sigma_v1(parent_1, parent_2)

	return child_genotype, child_sigma


def blended_crossover_v2(parent_1, parent_2):
	"""
	Performs recombination using the blended crossover methodology.
	Can choose between two sigma methods:
	- child_sigma_v1(parent_1, parent_2)
	- child_sigma_v2(parent_1, parent_2)
	"""
	difference = abs(parent_1.genotype - parent_2.genotype)

	# set alpha to 0.5 - equally likely to perform exploration and exploitation
	alpha = 0.5

	# sample random number uniformly from [0,1]
	mu = np.random.uniform(0,1)

	# calculate gamma (?)
	gamma = (1 - 2 * alpha) * mu - alpha

	# create child
	child_genotype  = (1 - gamma) * parent_1.genotype + gamma * parent_2

	# create child sigma
	child_sigma = child_sigma_v1(parent_1, parent_2)

	return child_genotype, child_sigma


def child_sigma_v1(parent_1, parent_2):
	"""
	Using this method, sigma cannot go below 0, but above parents' max sigmoids
	"""
	# sample random number uniformly from [0,1]
	mu = np.random.uniform(0,1)

	# take difference of parents' sigma values
	difference = max(abs(parent_1.sigma - parent_2.sigma))

	# sigma cannot be negative
	bound_1 = max(min(parent_1.sigma, parent_2.sigma) - mu * difference, 0)
	bound_2 = max(parent_1.sigma, parent_2.sigma) + mu * difference
	child_sigma = np.random.uniform(bound_1, bound_2)

	return child_sigma

def child_sigma_v2(parent_1, parent_2):
	"""
	Using this method sigma always stays between parents' sigmoids
	"""

	# sample random number uniformly from [0,1]
	mu = np.random.uniform(0,1)

	# let sigma stay between bounds [0,1]
	child_sigma = mu * parent_1.sigma + (1 - mu) * parent_2.sigma

	return child_sigma

def survival_selection(offspring, population_size):
    elitism = 0.1*len(offspring)
    leftover = population_size - elitism

    #The best 20 offspring always survives 
    best_offspring = list(sorted(offspring, key = lambda individual: individual.fitness)[-elitism:])

    #Pairwise tournament: the offspring with the higher fitness from the tournament survives.
    #This is repeated until we filled up the remaining "leftover" spots. 
    tournament_offspring = []
    
    while len(tournament_offspring) < leftover: 
        x = list(sorted(offspring, key = lambda individual: individual.fitness)[:leftover])
        potential1 = np.random.choice(x, 1, replace = False)
        potential2 = np.random.choice(x, 1, replace = False)
        
        #the tournament
        if potential1.fitness >= potential2.fitness:
            tournament_offspring.append(potential1)
        else:
            tournament_offspring.append(potential2)
        return tournament_offspring
    
    #constructing the new population consisting of these two groups of offspring
    new_population = best_offspring + tournament_offspring 
    return new_population

#note: we would have to delete the "population" input from the survival_selection in def generate_next_generation
#we would also have to make num_offspring = 2 in def generate_next_generation