##################################################################################
# This file contains the main methods for doing simulation, evolution, 
# implementing our two evolutioanry strategies.
##################################################################################


import numpy as np
from individual import Individual


def initialize_generation(environment, population_size, num_genes, adaptive_pop_size):
	"""
	Randomly initializes a generation by returning list of objects of class Individual.
	:param population_size: number of individuals in population
	:param num_genes: total number of weights in neural network controller
	"""
	# initialize all individuals in the population 
	all_genotypes = np.random.uniform(-1, 1, (population_size, num_genes))
	generation = [Individual(all_genotypes[i]) for i in range(population_size)]

	# compute fitness of all individuals
	for individual in generation:
		individual.fitness = individual.compute_fitness(environment)

	if adaptive_pop_size:
		assign_lifetime(generation)

	return generation


def generate_next_generation(environment, population, adaptive_pop_size, 
							 adaptive_mutation):
	"""
	Generates next generation from current population.
	:param environment: (simulation) environment object
	:param population: list of objects of class Individual
	"""
	# generate pairs of parents that can be used for recombination
	if adaptive_pop_size:
		parent_pairs = parent_selection_random(population, num_pairs=int(0.4*len(population)))
	else:
		parent_pairs = parent_selection_ranking(population, num_pairs=len(population)*4)

	# generate offspring
	offspring = []
	for i in range(len(parent_pairs)):
		children = create_offspring(environment, parent_pairs[i][0], parent_pairs[i][1], adaptive_mutation, num_offspring=1)
		offspring += children # concatenate children to offspring list	

	if adaptive_pop_size:
		assign_lifetime(offspring, population)
		
		new_population = offspring
		for individual in population:
			individual.lifetime -= 1
			if individual.lifetime > 0:
				new_population.append(individual)

		print([ind.lifetime for ind in new_population])
		
	else:
		# perform survival selection to return next generation with same size as input generation
		new_population = survival_selection(offspring, len(population))

	return new_population

def assign_lifetime(offspring, population=None):

	if population is None:
		population = offspring

	population_fitness = [ind.fitness for ind in population]
	offspring_fitness = [child.fitness for child in offspring]
	all_fitness = population_fitness + offspring_fitness
	pop_min = np.min(all_fitness)
	all_fitness = [value - pop_min for value in all_fitness]
	scaled_offspring_fitness = [value - pop_min for value in offspring_fitness]

	print(scaled_offspring_fitness)

	avg_fitness = np.mean(all_fitness)
	min_fitness = np.min(all_fitness)
	max_fitness = np.max(all_fitness)

	max_lt = 7
	min_lt = 1
	eta = (max_lt - min_lt) / 2

	for i, child in enumerate(offspring):
		if avg_fitness >= scaled_offspring_fitness[i]:
			offspring[i].lifetime = int(min_lt + eta * (scaled_offspring_fitness[i] - min_fitness) / (avg_fitness - min_fitness))
		else:
			offspring[i].lifetime = int((min_lt + max_lt)/2 + eta * (scaled_offspring_fitness[i]-avg_fitness) / (max_fitness - avg_fitness))

def parent_selection_random(population, num_pairs):
	parent_pairs = []
	while len(parent_pairs) != num_pairs:
		selection = np.random.choice(population, 2, replace=False)
		parent_pairs.append((selection[0], selection[1]))
	return parent_pairs

def parent_selection_ranking(population, num_pairs, s=1.5):
	"""
	Rank-based parent selection using linear ranking (with s = 1.5)
	:param: population: list of objects of class Individual
	:param num_pairs: number of parent pairs that we want to generate 
	"""
	# compute linearly adjusted ranks
	pop_size = len(population)
	sorted_population = sorted(population, key = lambda individual: individual.fitness)

	# compute linearly adjusted ranks
	selection_probs = []
	for i in range(len(sorted_population)):
		selection_probs.append(((2-s)/pop_size) + 2*i*(s-1)/(pop_size*(pop_size-1)))

	# sample random parent pairs
	parent_pairs = []
	while len(parent_pairs) != num_pairs:
		selection = np.random.choice(sorted_population, 2, replace=False, p=selection_probs)
		parent_pairs.append((selection[0], selection[1]))

	return parent_pairs


def create_offspring(environment, parent_1, parent_2, adaptive_mutation, num_offspring):
	"""
	Generate num_offspring from parent_1 and parent_2 using recombination and mutation
	:param environment: (simulation) environment object
	:param parent_1: first parent object of class Individual
	:param parent_2: first parent object of class Individual
	:param num_offspring: number of offspring to generate from the parent pair
	"""
	# create num_offspring children from the parent pair
	children = []
	for i in range(num_offspring):
		# apply whole arithmetic recombination to create children
		child = recombine(parent_1, parent_2)

		# apply mutation and add child to children list		
		if adaptive_mutation:
			child.mutate_self_adaptive()
		else:
			child.mutate()
	
		# compute child's fitness after mutation
		child.fitness = child.compute_fitness(environment)
		children.append(child)

	return children


def recombine(parent_1, parent_2):
	"""
	Performs recombination between two parents, creating a child.
	Use blend_crossover for ES1 and whole_arith_recombination for ES2.
	:param parent_1: first parent object of class Individual
	:param parent_2: first parent object of class Individual
	"""
	#child_genotype = blend_crossover(parent_1, parent_2)
	# use the following line to run ES2 vs. use the above line to run ES1
	child_genotype = whole_arith_recombination(parent_1, parent_2)

	# return new child object
	return Individual(child_genotype)


def whole_arith_recombination(parent_1, parent_2):
	"""
	This function applies whole arithmetic recombination (Eiben & Smith, 2015, p. 66)
	:param parent_1: first parent object of class Individual
	:param parent_2: first parent object of class Individual
	"""
	# compute child genotype
	alpha = np.random.uniform(0,1)
	child_genotype = alpha * parent_1.genotype + (1 - alpha) * parent_2.genotype

	return child_genotype


def blend_crossover(parent_1, parent_2):
	"""
	ref. A Crossover Operator Using Independent Component Analysis for Real-Coded Genetic Algorithms (Takahashi & Kita, 2001)
	"""
	alpha = 0.5 # ref Eshelmann & Schafer

	child_genotype = np.zeros((parent_1.num_genes,))
	for i in range(parent_1.num_genes):
		difference = abs(parent_1.genotype[i] - parent_2.genotype[i])
		bound_1 = min(parent_1.genotype[i], parent_2.genotype[i]) - alpha * difference
		bound_2 = max(parent_1.genotype[i], parent_2.genotype[i]) + alpha * difference
		child_genotype[i] = np.random.uniform(bound_1, bound_2)

	return child_genotype


def survival_selection(offspring, population_size):
	"""
	Perform survivor selection by picking best-performing offspring.
	"""
	sorted_offspring = sorted(offspring, key = lambda individual: individual.fitness)
	best_offspring = sorted_offspring[-population_size:]
	return best_offspring