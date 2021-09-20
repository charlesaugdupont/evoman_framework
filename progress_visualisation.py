import matplotlib.pyplot as plt
import numpy as np

def initialise_progress_plot():
    fig, axs = plt.subplots(3)
    fig.set_tight_layout(True)
    # interactive mode on
    plt.ion()
    return fig, axs

def plot_progress(fig, axs, population, fitness, generation):
    fig.suptitle(f"Information about generation {generation}")

    # visualise the population of the new generation
    y = sorted(fitness)
    axs[0].clear()
    axs[0].set_title("Fitness of Population")
    axs[0].set_xlabel("Individual")
    axs[0].set_ylabel("Fitness")
    axs[0].stem(y)

    # visualise genotypes
    weights = np.concatenate([individual.genotype[np.newaxis, :] for individual in population])
    axs[1].clear()
    axs[1].set_title("Genotypes")
    axs[1].set_xlabel("Gene")
    axs[1].set_ylabel("Individual")
    result = axs[1].matshow(weights)
    #fig.colorbar(result, ax = axs[1])

    # visualise step sizes
    step_sizes = np.concatenate([individual.sigma[np.newaxis, :] for individual in population])
    axs[2].clear()
    axs[2].set_title("Step sizes")
    axs[2].set_xlabel("Gene")
    axs[2].set_ylabel("Individual")
    result = axs[2].matshow(step_sizes)
    #fig.colorbar(result, ax = axs[2])

    # this is needed for some reason to make the plots work
    plt.pause(0.001)