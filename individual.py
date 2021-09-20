##################################################################################
# Python class for Individual.
# 
# An Individual belongs to a particular population for a given generation, and 
# has various attributes such as its genotypic representation and other
# information such as the sigma value (for mutation) or fitness.
##################################################################################

import numpy as np
from helpers import weight_limit

class Individual:
	def __init__(self, genotype, sigma):
		self.num_genes = genotype.shape[0]
		self.genotype = genotype
		self.sigma = sigma
		self.fitness = None # value is assigned by calling compute_fitness() from evolution.py 
		self.mutation_probability = 1.0
		# lower limit for sigma
		self.epsilon = 0.001
		# according to Eiben & Smith the common learning rate (tau') 
		# in self-adaptive mutation should be inversely proportional to double 
		# the square root of the genome size, and the gene-wise learning rate
		# should be inversely proportional to sqrt(2*sqrt(n)) (n = size of
		# genome)
		self.common_learning_rate = 2.0 * (1/np.sqrt(2*self.num_genes))
		self.gene_learning_rate = 2.0 * (1/np.sqrt(2*np.sqrt(self.num_genes)))

	def compute_fitness(self, environment):
		"""
		Computes fitness score and assigns it to fitness attribute.
		"""
		fitness,_,_,_ = environment.play(pcont=self.genotype)
		return fitness

	def mutate(self):
		"""
		Applies mutation to genotype.
		"""
		for i in range(0, self.num_genes):
			if np.random.uniform(0,1) <= self.mutation_probability:
				self.genotype[i] = self.genotype[i] + np.random.normal(0,1)
			self.genotype[i] = weight_limit(self.genotype[i])

	def mutate_self_adaptive(self):
		"""
		Applies uncorrelated mutation with n step sizes to genotype. Before
		this the step size is also mutated. 
		Mutation happens with probability 1, i.e. self.mutation_probability is 
		not enforced.
		"""
		# update sigma, first with overall learning rate, then gene-wise
		self.sigma = self.sigma * np.exp(np.random.normal(0, self.overall_learning_rate))
		self.sigma = self.sigma * np.exp(np.random.multivariate_normal(
			mean = np.zeros(self.num_genes),
			cov = self.gene_learning_rate * np.eye(self.num_genes)
		))
		# prevent the step size from going too small
		too_small = self.sigma < self.epsilon
		self.sigma[too_small] = self.epsilon
		# mutate with the updated step size
		for i in range(0, self.num_genes):
			self.genotype[i] = weight_limit(
				self.genotype[i] + np.random.multivariate_normal(
					mean = np.zeros(self.num_genes),
					cov = self.sigma * np.eye(self.num_genes)))
