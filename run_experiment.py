##################################################################################
# This file is the high level experiment script that we call from the command line
# e.g. "python run_experiment.py [arguments]"
#
# We can pass arguments such as enemy number, or experiment version.
#
# The goal of this script is to gather the data from all the runs and store it.
##################################################################################

# imports framework
import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
from nn_controller import player_controller

# other imports
import csv
import argparse
import numpy as np
from evolution import initialize_generation, generate_next_generation

import matplotlib.pyplot as plt

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# should progress be live-plotted
progress_visualisation = True
if progress_visualisation:
    from progress_visualisation import initialise_progress_plot, plot_progress


class Environment2(Environment):
    def cons_multi(self, values):
        return values.mean()


################################### SETTINGS

adaptive_mutation = True

###################################

# parses the arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument("--pop_size", type=int, required=False, default=30)
parser.add_argument("--num_gens", type=int, required=False, default=40)
parser.add_argument("--group", type=int, required=True)
parser.add_argument("--version", type=str, required=True)
args = parser.parse_args()

population_size = args.pop_size
num_generations = args.num_gens
group = int(args.group)
version = args.version

if group == 1:
    enemy_group = [7,8]
elif group == 2:
    enemy_group = [1,3]
else:
    print("ENEMY GROUP MUST BE 1 OR 2")
    sys.exit()

num_hidden_neurons = 10

# sets up experiment results folder for logs
experiment_name = os.path.join("experiment_results", version, "enemy_group_"+str(group))
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initialize environment
environment = Environment2(experiment_name=experiment_name,
                enemies=enemy_group,
                multiplemode="yes",
                playermode="ai",
                player_controller=player_controller(num_hidden_neurons),
                enemymode="static",
                randomini="yes",
                level=2,
                speed="fastest",
                logs="off") # avoid logging to stdout

# total number of "genes" or weights in the neural network controller
num_genes = (environment.get_num_sensors()+1)*num_hidden_neurons + (num_hidden_neurons+1)*5

if progress_visualisation:
    fig, axs = initialise_progress_plot()

# repeat experiment for a total of 10 runs
for run in range(10):
    print("\n--- SIMULATING RUN "+str(run+1) + " ---")

    # initialize first generation
    population = initialize_generation(environment, population_size, num_genes)

    # store best individual
    most_fit = max(population, key=lambda individual: individual.fitness)
    best_genotype, best_fitness = most_fit.genotype, most_fit.fitness
    np.savetxt(os.path.join(experiment_name, "run"+str(run)+"_best.txt"), best_genotype)

    # store overall generation metrics
    fitness_scores = [individual.fitness for individual in population]
    maximum, mean, sd = most_fit.fitness, np.mean(fitness_scores), np.std(fitness_scores)
    avg_step_size = np.mean([ind.sigma.mean() for ind in population])
    with open(os.path.join(experiment_name, "run"+str(run)+"_results.csv"), 'w', encoding="UTF-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation", "Max", "Mean", "SD", "Stepsize"]) # header
        print("Generation 0 : Max ({:.3f}) | Mean ({:.3f}) | SD ({:.3f}) | Avg step size ({:.3f})".format(maximum, mean, sd, avg_step_size)) # logging
        writer.writerow([0, maximum, mean, sd, avg_step_size])

    if progress_visualisation:
        plot_progress(fig, axs, population, fitness_scores, 0)

    # repeat num_generations times
    for iteration in range(1, num_generations+1):

        # evolve generation
        population = generate_next_generation(environment, population, adaptive_mutation)

        # update best genotype if needed
        most_fit = max(population, key=lambda individual: individual.fitness)
        if most_fit.fitness > best_fitness:
            best_genotype, best_fitness = most_fit.genotype, most_fit.fitness
            np.savetxt(os.path.join(experiment_name, "run"+str(run)+"_best.txt"), best_genotype)

        # store overall generation metrics
        fitness_scores = [individual.fitness for individual in population]
        maximum, mean, sd = most_fit.fitness, np.mean(fitness_scores), np.std(fitness_scores)
        avg_step_size = np.mean([ind.sigma.mean() for ind in population])
        clipped_prop = np.mean([np.mean((ind.genotype == -1.0) | (ind.genotype == 1.0)) for ind in population])
        with open(os.path.join(experiment_name, "run"+str(run)+"_results.csv"), 'a', encoding="UTF-8") as f:
            writer = csv.writer(f)
            writer.writerow([iteration, maximum, mean, sd, avg_step_size])
            print("Generation {} : Max ({:.3f}) | Mean ({:.3f}) | SD ({:.3f}) | Avg step size ({:.3f}) | Clipped prop ({:.2f})".format(iteration, maximum, mean, sd, avg_step_size, clipped_prop))

        if progress_visualisation:
            plot_progress(fig, axs, population, fitness_scores, iteration)