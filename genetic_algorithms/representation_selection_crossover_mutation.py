import numpy as np
import time
from math import sin, pi
# 목적 함수
def objective_function(x, y):
    return 21.5 + x * sin(4 * pi * x) + y * sin(20 * pi * y)
def decimal_to_binary(value, bounds, bits):
    lower, upper = bounds
    normalized = (value - lower) / (upper - lower)
    binary_value = format(int(normalized * (2 ** bits - 1)), f'0{bits}b')
    return binary_value
def binary_to_decimal(binary, bounds, bits):
    lower, upper = bounds
    decimal_value = int(binary, 2) / (2 ** bits - 1)
    return decimal_value * (upper - lower) + lower
# 초기 인구 생성 함수
def create_initial_population(pop_size, x_bits, y_bits, x_bounds, y_bounds):
    population = []
    for _ in range(pop_size):
        x_binary = decimal_to_binary(np.random.uniform(*x_bounds), x_bounds, x_bits)
        y_binary = decimal_to_binary(np.random.uniform(*y_bounds), y_bounds, y_bits)
        individual = x_binary + y_binary
        population.append(individual)
    return population
# 적합도 평가
def evaluate_population(population, x_bits, x_bounds, y_bounds):
    fitness = []
    for individual in population:
        x_binary, y_binary = individual[:x_bits], individual[x_bits:]
        x = binary_to_decimal(x_binary, x_bounds, x_bits)
        y = binary_to_decimal(y_binary, y_bounds, x_bits)
        fitness.append(objective_function(x, y))
    return fitness
# 선택 함수
def select(population, fitness, num_parents):
    parents = list(zip(population, fitness))
    parents.sort(key=lambda x: x[1], reverse=True)
    selected_parents = [ind[0] for ind in parents[:num_parents]]
    return selected_parents
# 교차 함수
def crossover(parents, offspring_size):
    offspring = []
    crossover_point = np.random.randint(1, offspring_size - 1)
    for k in range(offspring_size):
        parent1 = parents[k % len(parents)]
        parent2 = parents[(k + 1) % len(parents)]
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        offspring.extend([child1, child2])
    return offspring[:offspring_size]
# 변이
def mutate(offspring, mutation_rate=0.01):
    for idx in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            # 개별 유전자를 변이시키는 로직으로 수정
            mutation_point = np.random.randint(0, len(offspring[idx]))
            offspring[idx] = offspring[idx][:mutation_point] + \
                             ('1' if offspring[idx][mutation_point] == '0' else '0') + \
                             offspring[idx][mutation_point + 1:]
    return offspring
# 전체 유전 알고리즘
def genetic_algorithm(objective, x_bounds, y_bounds, pop_size=10, x_bits=16, y_bits=16, num_generations=1500,
                      num_parents=5):
    population = create_initial_population(pop_size, x_bits, y_bits, x_bounds, y_bounds)  # 수정된 부분
    best_outputs = []
    start_time = time.time()
    for generation in range(num_generations):
        fitness = evaluate_population(population, x_bits, x_bounds, y_bounds)  # 수정된 부분
        best_outputs.append(max(fitness))
        if generation in [499, 999, 1499]:
            print(
                f"Generation {generation + 1}: CPU Time = {time.time() - start_time}, Best Objective = {max(fitness)}")
        parents = select(population, fitness, num_parents)
        offspring_crossover = crossover(parents, offspring_size=pop_size - num_parents)
        offspring_mutation = mutate(offspring_crossover, mutation_rate=0.01)  # 수정이 필요한 부분
        population = parents + offspring_mutation
    return best_outputs
x_bounds = (-3.0, 12.1)
y_bounds = (4.1, 5.8)
results = genetic_algorithm(objective_function, x_bounds, y_bounds)
