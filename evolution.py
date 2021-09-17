##################################################################################
# This file could contain the main methods for doing simulation, evolution, 
# the genetic algorithm, etc.
##################################################################################

import numpy as np
from individual import Individual

def initialize_generation(environment, population_size, num_genes):
	"""
	Randomly initializes a generation by returning list of objects of class Individual.
	:param population_size: number of individuals in population
	:param num_genes: total number of weights in neural network controller
	"""
	all_genotypes = np.random.uniform(-1, 1, (population_size, num_genes))
	all_sigmas = np.random.uniform(0.1, 1.0, (population_size,))
	generation = [Individual(environment, all_genotypes[i], all_sigmas[i]) for i in range(population_size)]
	return generation

def generate_next_generation(environment, population):
	"""
	Generates next generation from current population.
	:param population: list of objects of class Individual
	"""
	# generate pairs of parents that can be used for recombination
	parent_pairs = parent_selection(population)

	# generate offspring
	offspring = []
	for i in range(len(parent_pairs)):
		children = create_offspring(environment, parent_pairs[i][0], parent_pairs[i][1], num_offspring=1)
		offspring += children # concatenate children to offspring list

	# perform survival selection to return next generation with same size as input generation
	new_population = survival_selection(population, offspring)

	return new_population

def parent_selection(population):
	"""
	Returns a list of 2-tuples with size |population| containing selected parent genotype vectors.
	:param population: list of objects of class Individual
	"""
	# compute desired number of parent pairings
	num_pairs = int(len(population)/2)

	# iterate until we have as many parent pairs as the size of our population
	parent_list = []
	while len(parent_list) != num_pairs:
		# draw random sample of 5 individuals
		sample = np.random.choice(population, 5, replace=False)
		# select 2 parents with highest fitness scores
		top_2 = sorted(sample, key=lambda individual: individual.fitness)[-2:]
		parent_list.append((top_2[0], top_2[1]))

	return parent_list

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
		child = recombine(environment, parent_1, parent_2)
		# apply mutation and add child to children list		
		child.mutate()
		children.append(child)

	return children

def recombine(environment, parent_1, parent_2):
	"""
	Performs recombination between two parents, creating a child.
	"""
	# compute child genotype
	alpha = np.random.uniform(0,1)
	child_genotype = alpha * parent_1.genotype + (1 - alpha) * parent_2.genotype

	# compute child sigma (TODO : NEEDS TO BE CHANGED)
	child_sigma = np.random.uniform(0.1, 1.0)

	# return new child object
	return Individual(environment, child_genotype, child_sigma)

def survival_selection(population, offspring):
	"""
	Choose which individuals from the (parent) population and from the offspring 
	survive to the next generation.

	Currently, all the offspring are selected, and the remaining spots are filled from the population based
	on fitness from highest to lowest. It is assumed that the number of offspring is no more
	than the size of the population.

	:param population: list of objects of class Individual representing the parent (current) population
	:param offspring: list of objects of class Individual representing the generated offspring
	:return value: list of objects of class Individual with length equal to size of parent population 
	"""
	# compute how many parents we will select to survive
	num_parents = len(population) - len(offspring)

	# construct new population consisting of ALL offspring, and the most fit parents for the remaining spots
	new_population = offspring + sorted(population, key=lambda individual: individual.fitness)[-num_parents:]

	return new_population