##################################################################################
# This file could contain the main methods for doing simulation, evolution, 
# the genetic algorithm, etc.
##################################################################################

import numpy as np


# constants to use, for experimentation now:
num_genes = 265
fitness_population = np.array() # of fitness values of population

def initialize_generation(population_size, num_genes):
	"""
	Randomly initializes a generation.
	:param population_size: number of individuals in population
	:param num_genes: total number of weights in neural network controller
	"""
	return np.random.uniform(-1, 1, (population_size, num_genes))


def generate_next_generation(population, population_fitness):
	"""
	Nils
	Generates next generation from current population.
	:param population_fitness: array containing each individual in population and their fitness score
	"""

	# generate pairs of parents that can be used for recombination
	parent_pairs = parent_selection(population, population_fitness)

	# generate offspring
	for i in len(parent_pairs):
		children, children_fitness = recombine(parent_pairs[i][0], parent_pairs[i][1])
	

	# population and population fitness arrays for next generation
	new_population, new_population_fitness = survivor_selection(population, population_fitness, children, children_fitness)

	return new_population, new_population_fitness

def parent_selection(population_array, fitness_array):
	"""
	Charles
	Returns a list of pairs with size |population| containing the selected parents that will reproduce.
	:param populaton_fitness: array containing each individual in population and their fitness score
	"""
	return

def recombine(parent_1, parent_2, num_genes, num_offspring, environment):
	"""
	Nils
	Generate one offspring from individuals x and y using crossover
	:param parent_1: numpy vector with 265 weights
	:param parent_2: numpy vector with 265 weights
	:param num_genes: number of weights in vector for child
	:param num_offspring: number of offspring to generate
	:param environment: environment in which recombination happens
	This function applies whole arithmetic recombination (Eiben & Smith, 2015, p. 66)
	"""

	# generate desired number of children
	children = np.zeros((num_offspring, num_genes))
	children_fitness = np.zeros((num_offspring, 1))

	# Apply whole arithmetic recombination to create children
	for i in len(children):
		child = alpha * parent_1 + (1 - alpha) * parent_2

		# normalize each new vector between [-1, 1]
		# put this in separate function, and choose a more appropriate scaling mechanism

		# apply mutation on each child
		mutate(child)

		# normalize each new vector between [-1, 1]
		# put this in separate function, and choose a more appropriate scaling mechanism
		for j in child:
			if j > upper_bound:
				j = upper_bound
			elif j < lower_bound:
				j = lower_bound

		# set child in array		
		children[i] = child

		# set fitness score in array
		children_fitness[i] = compute_fitness(environment, children[i])

	return children, children_fitness

def mutate(individual):
	"""
	Johanna
	Applies random mutation to individual.
	:param x: numpy vector with 265 weights
	"""
	return

def compute_fitness(environment, individual):
	"""
	Evaluate the fitness of individual x.
	:param x: individual x
	"""
	fitness, player_life, enemy_life, time = environment.play(pcont=individual)
	return fitness

def survival_selection():
	"""
	Otto
	"""
	return