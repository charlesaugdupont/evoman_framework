##################################################################################
# This file is the main experiment script that we call from the command line
# e.g. "python run_experiment.py [arguments]"
#
# We can pass arguments such as population size, number of generations,
# enemy number, even perhaps which EA to use etc...
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

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# should progress be live-plotted
progress_visualisation = False
if progress_visualisation:
    from progress_visualisation import initialise_progress_plot, plot_progress

# parses the arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument("--pop_size", type=int, required=False, default=30)
parser.add_argument("--num_gens", type=int, required=False, default=30)
parser.add_argument("--num-neurons", type=int, required=False, default=10)
parser.add_argument("--algorithm", type=str, required=False, default="GA")
parser.add_argument("--enemy", type=int, required=True)
parser.add_argument("--version", type=str, required=True)
args = parser.parse_args()

population_size = args.pop_size
num_generations = args.num_gens
num_hidden_neurons = args.num_neurons
enemy = args.enemy
algorithm = args.algorithm
version = args.version

# sets up experiment results folder for logs
experiment_name = os.path.join("experiment_results", algorithm, version, "enemy"+str(enemy))
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initialize environment
environment = Environment(experiment_name=experiment_name,
                enemies=[enemy],
                playermode="ai",
                player_controller=player_controller(num_hidden_neurons),
                enemymode="static",
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
    with open(os.path.join(experiment_name, "run"+str(run)+"_results.csv"), 'w', encoding="UTF-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation", "Max", "Mean", "SD"]) # header
        print("Generation 1 : Max ({:.3f}) | Mean ({:.3f}) | SD ({:.3f})".format(maximum, mean, sd)) # logging
        writer.writerow([0, maximum, mean, sd])

    if progress_visualisation:
        plot_progress(fig, axs, population, fitness_scores, 0)

    # repeat num_generations times
    for iteration in range(1, num_generations):

        # evolve generation
        population = generate_next_generation(environment, population)

        # update best genotype if needed
        most_fit = max(population, key=lambda individual: individual.fitness)
        if most_fit.fitness > best_fitness:
            best_genotype, best_fitness = most_fit.genotype, most_fit.fitness
            np.savetxt(os.path.join(experiment_name, "run"+str(run)+"_best.txt"), best_genotype)

        # store overall generation metrics
        fitness_scores = [individual.fitness for individual in population]
        maximum, mean, sd = most_fit.fitness, np.mean(fitness_scores), np.std(fitness_scores)
        with open(os.path.join(experiment_name, "run"+str(run)+"_results.csv"), 'a', encoding="UTF-8") as f:
            writer = csv.writer(f)
            writer.writerow([iteration, maximum, mean, sd])
            print("Generation {} : Max ({:.3f}) | Mean ({:.3f}) | SD ({:.3f})".format(iteration+1, maximum, mean, sd))

        if progress_visualisation:
            plot_progress(fig, axs, population, fitness_scores, iteration)