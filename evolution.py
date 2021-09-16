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

def surivival_selection():
	"""
	Otto
	"""