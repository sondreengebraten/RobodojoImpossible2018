import random

from deap import base
from deap import creator
from deap import tools

import numpy as np
import matplotlib.pyplot as plt

from math import sin, cos

def generate_cities(n):
    cities = []

    for i in range(n):
        cities.append((random.random(), random.random()))

    return 0, cities

def generate_cities_clusters(n):
    cities = []

    num_clusters = 6
    for c in range(num_clusters):
        cc = np.array((random.random(), random.random()))
        for i in range(n/num_clusters):
            pos = np.array((random.random()*0.1, random.random()*0.1)) + cc
            cities.append(pos)

    _,extra_cities = generate_cities(n-len(cities))
    cities.extend(extra_cities)

    return 0, cities

def generate_cities_circle(n):
    cities = []

    for i in range(n):
        cities.append(np.array((cos(float(i)*6.28/n),(sin(float(i)*6.28/n)))))

    return evaluate_sequence(cities), cities

def make_sequence(cities, individual):
    priorities = list(individual)

    t = zip(priorities, cities)

    t = sorted(t, key=lambda v: v[0])

    return map(lambda x: x[1], t)

def distance(pos1,pos2):
    return np.linalg.norm(np.array(pos1)-np.array(pos2))

def evaluate_sequence(sequence):

    acc_distance = 0.

    for i in range(len(sequence)-1):
        acc_distance += distance(sequence[i], sequence[i+1])

    acc_distance += distance(sequence[0], sequence[-1])

    return acc_distance

def factorial(n):
    if n > 1:
        return factorial(n-1)*n
    else:
        return 1

def plot_figures(g, cities, best_ind, fitness_time):
    fig = plt.figure("Cities")
    plt.ion()
    plt.clf()
    plt.xlim(-0.1,1.1)
    plt.ylim(-0.1,1.1)

    num_cities = len(cities)
    
    sequence = make_sequence(cities, best_ind)

    xs = map(lambda v: v[0], sequence)
    ys = map(lambda v: v[1], sequence)

    plt.scatter(xs,ys)

    xs.append(sequence[0][0])
    ys.append(sequence[0][1])

    plt.plot(xs,ys)

    xmin, xmax = plt.gca().get_xlim()
    ymin, ymax = plt.gca().get_ylim()
    plt.title("Prob %s. Gen %s. Fit %s. " % (num_cities, g, -round(best_ind.fitness.values[0],2)), fontsize="14")

    plt.gca().set_aspect("equal")

    fig.canvas.draw()


    fig = plt.figure("Fitness")
    plt.clf()

    plt.xlabel("Generation")
    plt.ylabel("Fitness")


    fitness_time.append(-best_ind.fitness.values[0])
    plt.plot(fitness_time)

    fig.canvas.draw()

    plt.show()

def main():
    # the goal ('fitness') function to be maximized
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # register the crossover operator
    toolbox.register("mate", tools.cxOnePoint)

    # register a mutation operator with a probability to
    # flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.4, indpb=0.5)

    # operator for selecting individuals for breeding the next generation
    #toolbox.register("select", tools.selRoulette)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def checkBounds(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in xrange(len(child)):
                    if child[i] > 1.:
                        child[i] = 1.
                    elif child[i] < 0.:
                        child[i] = 0.
            return offspring
        return wrapper

    toolbox.decorate("mate", checkBounds)
    toolbox.decorate("mutate", checkBounds)

    random.seed(64)


    elitism = 2

    for num_cities in range(35,100):
        optimal_solution, cities = generate_cities_clusters(num_cities)

        #Function to evaluate the fitness of an individual
        def evaluate(individual):
            sequence = make_sequence(cities, individual)
            fitness = evaluate_sequence(sequence)

            return -fitness,

        # Attribute generator 
        toolbox.register("attr_weight", random.random, )

        # Structure initializers
        toolbox.register("individual", tools.initRepeat, creator.Individual, 
            toolbox.attr_weight, num_cities)

        # define the population to be a list of individuals
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # register the goal / fitness function
        toolbox.register("evaluate", evaluate)

        # create an initial population

        population_size = 1000# 10* num_cities**2
        pop = toolbox.population(n=population_size)

        # CXPB  is the probability with which two individuals
        #       are crossed
        #
        # MUTPB is the probability for mutating an individual
        CXPB, MUTPB = 0.5, 0.5


        # Variable keeping track of the number of generations
        g = 0
        
        # Evaluate the entire population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Begin the evolution

        fitness_time = []
        while g < 100:

            # A new generation
            g = g + 1
            
            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))
        
            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                # cross two individuals with probability CXPB
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)

                    # fitness values of the children
                    # must be recalculated later
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:

                # mutate an individual with probability MUTPB
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
        
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            

            elites  = tools.selBest(pop, elitism)

            #Population is replace by the offspring + selected elites
            pop[:elitism] = elites
            pop[elitism:] = offspring[elitism:]

            best_ind = tools.selBest(pop, 1)[0]
            plot_figures(g, cities, best_ind, fitness_time)

            if abs(optimal_solution+best_ind.fitness.values[0]) < 0.01: 
                break

            # Gather all the fitnesses in one list and print the stats
            #fits = [ind.fitness.values[0] for ind in pop]
            
            # length = len(pop)
            # mean = sum(fits) / length
            # sum2 = sum(x*x for x in fits)
            # std = abs(sum2 / length - mean**2)**0.5
            
            # print("  Min %s" % min(fits))
            # print("  Max %s" % max(fits))
            # print("  Avg %s" % mean)
            # print("  Std %s" % std)

if __name__ == "__main__":
    main()
