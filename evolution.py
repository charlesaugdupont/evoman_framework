##################################################################################
# This file could contain the main methods for doing simulation, evolution, 
# the genetic algorithm, etc.
##################################################################################

import numpy as np

def initialize_generation(population_size, num_genes):
	"""
	Randomly initializes a generation.
	:param population_size: number of individuals in population
	:param num_genes: total number of weights in neural network controller
	"""
	return np.random.uniform(-1, 1, (population_size, num_genes))


def generate_next_generation(population_array, fitness_array):
	"""
	Nils
	Generates next generation from current population.
	:param population_fitness: array containing each individual in population and their fitness score
	"""
	# should normalize each new vector between [-1, 1]
	return

def parent_selection(population_array, fitness_array):
	"""
	Charles
	Returns a list of pairs with size |population| containing the selected parents that will reproduce.
	:param populaton_fitness: array containing each individual in population and their fitness score
	"""
	return

def recombine(parent_1, parent_2):
	"""
	Nils
	Generate one offspring from individuals x and y using crossover.
	:param x: first individual ; numpy vector with 265 weights
	:param y: second individual ; numpy vector with 265 weights
	"""
	return

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

def survival_selection(population, population_fitness, offspring, offspring_fitness):
	"""
	Choose which individuals from the (parent) population and from the offspring 
	survive to the next generation.

	Currently, all the offspring are selected, and the remaining spots are filled from the population based
	on fitness from highest to lowest. It is assumed that the number of offspring is no more
	than the size of the population.

	:param population_array: The (parent) population. An array of shape (population_size, num_genes)
	:param fitness_array: Fitness of the population. A 1-d array of length population_size
	:param offspring_array: The offspring generated from the parent population. An array of shape 
	(num_offspring, num_genes)
	:param offspring_fitness_array: The fitness of the offspring. A 1-d array of length num_offspring
	:return value: A tuple (population, fitness) where the population is an array of shape 
	(population_size, num_genes) and fitness is a 1-d array of length population_size
	"""
	# we choose all the offspring
	new_population = offspring
	new_fitness = offspring_fitness

	# how many parents we will select to survive
	num_parents = population.shape[0] - offspring.shape[0]

	# rankings of the parent population from lowest to highest fitness
	parent_ranking = np.argsort(population_fitness)
	# choose num_parents best from the parents
	parent_ranking = parent_ranking[-num_parents:]
	new_population = np.vstack((new_population, population[parent_ranking,:]))
	new_fitness = np.append(new_fitness, population_fitness[parent_ranking])
	
	return new_population, new_fitness

# testing the function
# parents = np.random.uniform(-1, 1, size = (10, 3))
# print(parents)
# fitness = np.random.normal(0, 100, size = (10,))
# print(fitness)
# offspring = np.random.uniform(-1, 1, size = (5, 3))
# print(offspring)
# offspring_fitness = np.random.normal(0, 100, size = (5,))
# print(offspring_fitness)
# new_population, new_fitness = survival_selection(parents, fitness, offspring, offspring_fitness)
# print(new_population)
# print(new_fitness)

