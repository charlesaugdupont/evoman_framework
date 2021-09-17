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
from evolution import initialize_generation, generate_next_generation, compute_fitness

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# parses the arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument("--pop_size", type=int, required=False, default=100)
parser.add_argument("--num_gens", type=int, required=False, default=30)
parser.add_argument("--num-neurons", type=int, required=False, default=10)
parser.add_argument("--enemy", type=int, required=True)
args = parser.parse_args()

population_size = args.pop_size
num_generations = args.num_gens
num_hidden_neurons = args.num_neurons
enemy = args.enemy

# sets up experiment results folder for logs
experiment_name = "experiment_results/"+"enemy"+str(enemy)
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

# repeat experiment for a total of 10 runs
for run in range(10):

    # initialize first generation : numpy array with dimension (100, 265)
    population_array = initialize_generation(population_size, num_genes)
    # compute fitness scores
    fitness_array = np.zeros((population_size,))
    for index, individual in enumerate(population_array):
        fitness_array[index] = compute_fitness(environment, individual)

    # store generation 0 metrics
    with open(os.path.join(experiment_name, "run"+str(run)+"_results.csv"), 'w', encoding="UTF-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation", "Max", "Mean", "SD"]) # header
        writer.writerow([0, np.max(fitness_array), np.mean(fitness_array), np.std(fitness_array)])
    
    # store best individual
    max_fitness_index = np.argmax(fitness_array)
    best_individual, best_fitness = population_array[max_fitness_index], fitness_array[max_fitness_index]
    np.savetxt(os.path.join(experiment_name, "run"+str(run)+"_best.txt"), best_individual)

    # repeat num_generations times
    for iteration in range(1, num_generations):
        # evolve generation
        population_array, fitness_array = generate_next_generation(environment, population_array, fitness_array)
        max_fitness_index = np.argmax(fitness_array)
        with open(os.path.join(experiment_name, "run"+str(run)+"_results.csv"), 'a', encoding="UTF-8") as f:
            writer = csv.writer(f)
            writer.writerow([iteration, fitness_array[max_fitness_index], np.mean(fitness_array), np.std(fitness_array)])
        if fitness_array[max_fitness_index] > best_fitness:
            best_individual, best_fitness = population_array[max_fitness_index], fitness_array[max_fitness_index]
            np.savetxt(os.path.join(experiment_name, "run"+str(run)+"_best.txt"), best_individual)