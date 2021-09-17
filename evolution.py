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

def parent_selection(population, population_fitness):
	"""
	Returns a list of 2-tuples with size |population| containing selected parent genotype vectors.
	:param population: numpy array containing each individual in population
	:param population_fitness: numpy array containing each individual's fitness score
	"""
	parent_list = []
	population_size = population.shape[0]
	# iterate until we have as many parent pairs as the size of our population
	while len(parent_list) != population_size:
		# draw random sample of 5 individuals
		sample = np.random.choice(population_size, 5, replace=False)
		# select 2 parents with highest fitness scores
		top_2 = sorted(sample, key=lambda index : population_fitness[index])[-2:]
		parent_list.append((population[top_2[0]], population[top_2[1]]))
	return parent_list

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

