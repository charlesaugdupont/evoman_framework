##################################################################################
# Python class for Individual.
# 
# An Individual belongs to a particular population for a given generation, and 
# has various attributes such as its genotypic representation and other
# information such as its fitness, mutation probability for its genes etc.
##################################################################################


import numpy as np
from helpers import weight_limit


class Individual:

	def __init__(self, genotype):
		"""
		Creates object of class Individual.
		:param genotype: numpy vector of length 265.
		"""
		self.num_genes = genotype.shape[0]
		self.genotype = genotype
		self.fitness = None # value is assigned by calling compute_fitness() from evolution.py 
		self.mutation_probability = 0.7
		self.lifetime = 0

	def compute_fitness(self, environment):
		"""
		Computes fitness score and assigns it to fitness attribute.
		:param environment: (simulation) environment object
		"""
		fitness, player_energy, enemy_energy, time = environment.play(pcont=self.genotype)
		return player_energy-enemy_energy # gain

	def mutate(self):
		"""
		Applies mutation to genotype by introducing gaussian noise
		from a normal distribution with mean 0 and step size 0.1.
		The mutation probability is set to 0.7. 
		"""
		for i in range(0, self.num_genes):
			if np.random.uniform(0,1) <= self.mutation_probability:
				self.genotype[i] = self.genotype[i] + np.random.normal(0,0.1)
			self.genotype[i] = weight_limit(self.genotype[i])