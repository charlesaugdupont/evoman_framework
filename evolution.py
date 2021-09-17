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


def weightlimit(w):
	"""
	Johanna 
	defining upper and lower limits for the weights in the individual's array.  
	"""
	upper_weightlimit= 1
	lower_weightlimit = -1

	if w > upper_weightlimit:
		return upper_weightlimit 
	elif w < lower_weightlimit:
		return lower_weightlimit 
	else:
		return w 

def mutate(individual):
	"""
	Johanna
	:param individual: numpy vector with 265 weights
	:param mutation_probability: for each weight within an individual's vector, there is a 20% chance that a random mutation is applied.
	- the mutation size is drawn from a normal distribution (with mean = 0 and std = 1)
	i: iterating through each weight of an individual's vector  
	"""
	mutation_probability = 0.2
	for i in range(0, len(individual)):
		if np.random.uniform(0,1) <= mutation_probability:
			individual[i] = individual[i] + np.random.normal(0,1)

	individual = np.array(list(map(lambda y: weightlimit(y), individual))) #iterating through the weights of a mutated individual to make sure they are still between [-1, 1]. 
	return individual

def compute_fitness(environment, individual):
	"""
	Evaluate the fitness of individual x.
	:param x: individual x
	"""
	fitness, player_life, enemy_life, time = environment.play(pcont=individual)
	return fitness

def surivival_selection():
	"""
	Otto
	"""