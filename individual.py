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

	def __init__(self, genotype, sigma):
		"""
		Creates object of class Individual.
		:param genotype: numpy vector of length 265.
		"""
		self.num_genes = genotype.shape[0]
		self.genotype = genotype
		self.sigma = sigma
		self.fitness = None # value is assigned by calling compute_fitness() from evolution.py 
		self.mutation_probability = 1.0
		self.lifetime = 0
		# lower limit for sigma
		self.epsilon = 0.001
		# according to Eiben & Smith the common learning rate (tau') 
		# in self-adaptive mutation should be inversely proportional to double 
		# the square root of the genome size, and the gene-wise learning rate
		# should be inversely proportional to sqrt(2*sqrt(n)) (n = size of
		# genome)
		self.common_learning_rate = 2 * (1/np.sqrt(2*self.num_genes))
		self.gene_learning_rate = 2 * (1/np.sqrt(2*np.sqrt(self.num_genes)))

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

	def mutate_self_adaptive(self):
		"""
		Applies uncorrelated mutation with n step sizes to genotype. Before
		this the step sizes are also mutated.
		Each gene is mutated with probability self.mutation_probability.
		"""
		# update sigma, first with common learning rate, then gene-wise
		self.sigma = self.sigma * np.exp(np.random.normal(0, self.common_learning_rate))
		# gene-wise update
		self.sigma = self.sigma * np.exp(np.random.normal(
			loc = 0,
			scale = self.gene_learning_rate,
			size =  (self.num_genes,)
		))

		# prevent the step size from going too small
		self.sigma[self.sigma < self.epsilon] = self.epsilon

		# which genes to mutate
		genes_to_mutate = np.random.uniform(0,1,self.num_genes) <= self.mutation_probability
		# "mask" actual mutation amounts with genes_to_mutate
		#mutation_amount = genes_to_mutate * np.random.multivariate_normal(
		#	mean = np.zeros(self.num_genes),
		#	cov = np.diag(self.sigma**2))

		# calculate mutation amounts individually to make sure it goes right
		# this time
		mutation_amount = np.zeros((self.num_genes,))
		for i in range(self.num_genes):
			mutation_amount[i] = genes_to_mutate[i] * np.random.normal(0, self.sigma[i])

		# apply mutation to genotype
		self.genotype = weight_limit(self.genotype + mutation_amount)
