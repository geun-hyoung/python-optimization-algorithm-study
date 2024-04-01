import numpy as np
import time
from math import sin, pi

# 목적 함수 정의
def objective_function(x, y):
    return 21.5 + x * sin(4 * pi * x) + y * sin(20 * pi * y)

# 십진수를 이진수로 변환
def decimal_to_binary(value, bounds, bits):
    lower, upper = bounds
    # 십진수 값을 [0, 1] 범위로 정규화
    normalized = (value - lower) / (upper - lower)
    # 정규화된 값을 이진수로 변환
    binary_value = format(int(normalized * (2**bits - 1)), f'0{bits}b')
    return binary_value

# 이진수를 십진수로 변환
def binary_to_decimal(binary, bounds, bits):
    lower, upper = bounds
    decimal_value = int(binary, 2) / (2**bits - 1)
    # 이진수 값을 원래 범위로 변환
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

# 적합도 평가 함수
def evaluate_population(population, x_bits, x_bounds, y_bounds):
    fitness = []
    for individual in population:
        x_binary, y_binary = individual[:x_bits], individual[x_bits:]
        x = binary_to_decimal(x_binary, x_bounds, x_bits)
        y = binary_to_decimal(y_binary, y_bounds, x_bits)
        fitness.append(objective_function(x, y))
    return fitness

# 선택, 교차, 변이 함수는 이진 표현 방식에 맞게 동일하게 적용됩니다.

# 유전 알고리즘 실행 함수는 생략됩니다. (위의 코드 참조)

# 유전 알고리즘 실행
x_bounds = (-3.0, 12.1)
y_bounds = (4.1, 5.8)
x_bits = 16  # x 변수를 표현하기 위한 비트 수
y_bits = 16  # y 변수를 표현하기 위한 비트 수

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
        parent2 = parents[(k+1) % len(parents)]
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        offspring.extend([child1, child2])
    return offspring[:offspring_size]

# 변이 함수
def mutate(offspring, x_bounds, y_bounds, mutation_rate=0.01):
    for idx in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            offspring[idx] = (np.random.uniform(*x_bounds), np.random.uniform(*y_bounds))
    return offspring

# 유전 알고리즘 실행 함수
def genetic_algorithm(objective, x_bounds, y_bounds, pop_size=10, num_generations=1500, num_parents=5):
    population = create_initial_population(pop_size, x_bounds, y_bounds)
    best_outputs = []
    start_time = time.time()

    for generation in range(num_generations):
        fitness = evaluate_population(population)
        best_outputs.append(max(fitness))
        if generation in [499, 999, 1499]:
            print(f"Generation {generation + 1}: CPU Time = {time.time() - start_time}, Best Objective = {max(fitness)}")
        parents = select(population, fitness, num_parents)
        offspring_crossover = crossover(parents, offspring_size=pop_size - num_parents)
        offspring_mutation = mutate(offspring_crossover, x_bounds, y_bounds)
        population = parents + offspring_mutation

    return best_outputs

