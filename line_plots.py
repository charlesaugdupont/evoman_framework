import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--group", type=int, required=True)
parser.add_argument("--version", type=str, required=True)
args = parser.parse_args()
group = int(args.group)
version = args.version
experiment_name = os.path.join("experiment_results", version, "enemy_group_"+str(group))

# read data
mean_fitness, max_fitness = [], []
for f in os.listdir(experiment_name):
	if 'results.csv' in f:
		df = pd.read_csv(os.path.join(experiment_name, f))
		mean_fitness.append(list(df.Mean))
		max_fitness.append(list(df.Max))

# compute means and standard deviations of maximum and average fitness scores for each generation across all runs
max_fitness = np.vstack(max_fitness)
max_fitness_mean = np.mean(max_fitness, axis=0)
max_fitness_sd = np.std(max_fitness, axis=0)

mean_fitness = np.vstack(mean_fitness)
mean_fitness_mean = np.mean(mean_fitness, axis=0)
mean_fitness_sd = np.std(mean_fitness, axis=0)
num_generations = len(mean_fitness_mean)

# create plot
fig = plt.figure()
plt.plot(np.arange(0, num_generations), mean_fitness_mean, 'r', label="mean")
plt.fill_between(np.arange(0, num_generations), mean_fitness_mean-mean_fitness_sd, mean_fitness_mean+mean_fitness_sd, color='salmon', alpha=0.5)
plt.plot(np.arange(0, num_generations), max_fitness_mean, 'b', label="max")
plt.fill_between(np.arange(0, num_generations), max_fitness_mean-max_fitness_sd, max_fitness_mean+max_fitness_sd, color='skyblue', alpha=0.5)
plt.xlabel("Generation")
plt.ylabel("Fitness")

plt.grid()
plt.legend()
fig.savefig(os.path.join(experiment_name, "line_plot.png"), dpi=fig.dpi)