import numpy as np
from math import sin, pi
import time


def objective_function(x, y):
    # numpy 배열을 받아서 연산 가능하도록 수정
    return 21.5 + x * np.sin(4 * np.pi * x) + y * np.sin(20 * np.pi * y)

def create_population(pop_size, x_bounds, y_bounds, bits=16):
    x_population = np.random.uniform(x_bounds[0], x_bounds[1], pop_size)
    y_population = np.random.uniform(y_bounds[0], y_bounds[1], pop_size)
    return np.column_stack((x_population, y_population))


def evaluate_population(population):
    x, y = population[:, 0], population[:, 1]
    return objective_function(x, y)


def select(population, fitness, num_parents):
    idx = np.argsort(fitness)[::-1][:num_parents]
    return population[idx]


def crossover(parents, offspring_size):
    offspring = np.empty((offspring_size, parents.shape[1]))
    crossover_point = np.random.randint(1, offspring_size - 1)

    for k in range(offspring_size):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


def mutate(offspring, mutation_rate=0.01):
    mutations = np.random.rand(offspring.shape[0], offspring.shape[1]) < mutation_rate
    random_values = np.random.uniform(-1, 1, offspring.shape)
    offspring += mutations * random_values
    return offspring


def genetic_algorithm(objective, x_bounds, y_bounds, pop_size=10, num_generations=1500, num_parents=5):
    population = create_population(pop_size, x_bounds, y_bounds)
    best_outputs = []
    start_time = time.time()

    for generation in range(num_generations):
        fitness = evaluate_population(population)
        best_outputs.append(np.max(fitness))

        if generation in [499, 999, 1499]:
            print(
                f"Generation {generation + 1}: CPU Time = {time.time() - start_time}, Best Objective = {np.max(fitness)}")

        parents = select(population, fitness, num_parents)
        offspring_crossover = crossover(parents, offspring_size=pop_size - num_parents)
        offspring_mutation = mutate(offspring_crossover)
        population = np.vstack((parents, offspring_mutation))

    return best_outputs


x_bounds = (-3.0, 12.1)
y_bounds = (4.1, 5.8)
results = genetic_algorithm(objective_function, x_bounds, y_bounds)


