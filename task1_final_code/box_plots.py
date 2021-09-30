import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
from nn_controller import player_controller

import matplotlib.pyplot as plt
import numpy as np
import argparse

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--enemy", type=int, required=True)
parser.add_argument("--algorithm", type=str, required=False, default="GA")
parser.add_argument("--num-neurons", type=int, required=False, default=10)
parser.add_argument("--version", type=str, required=True)
args = parser.parse_args()
enemy = args.enemy
algorithm = args.algorithm
version = args.version
num_hidden_neurons = int(args.num_neurons)
experiment_name = os.path.join("experiment_results", algorithm, version, "enemy"+str(enemy))

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
	os.environ["SDL_VIDEODRIVER"] = "dummy"

# initialize environment
environment = Environment(experiment_name=experiment_name,
				enemies=[enemy],
				playermode="ai",
				player_controller=player_controller(num_hidden_neurons),
				enemymode="static",
				level=2,
				speed="fastest", 
				logs="off") # avoid logging to stdout

# read data
best_solutions = []
for f in os.listdir(experiment_name):
	if 'best.txt' in f:
		best_solutions.append(np.loadtxt(os.path.join(experiment_name, f)))

# simulate each solution 5 times and compute average gains
means = []
for index, solution in enumerate(best_solutions):
	print("\n- SOLUTION "+str(index+1) + " -")
	total_gain = 0
	for trial in range(5):
		fitness, player_energy, enemy_energy, time = environment.play(pcont=solution)
		gain = player_energy - enemy_energy
		print("Run {} Gain = {}".format(trial+1, gain))
		total_gain += gain
	means.append(total_gain/5)

# create plot
fig = plt.figure()
plt.boxplot(means, labels=["EA1"])
plt.grid()
plt.ylabel("Gain")
plt.title("Performance of Best Solutions (Enemy {})".format(enemy))
fig.savefig(os.path.join(experiment_name, "box_plot.png"), dpi=fig.dpi)