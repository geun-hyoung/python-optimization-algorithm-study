import math
import random
import numpy as np

def object_function(x, y):
    return (4 - 2.1 * x ** 2 + (x ** 4) / 3) * x ** 2 + x * y + (-4 + 4 * y ** 2) * y ** 2

def binary_to_float(binary, x_bounds, y_bounds, x_bits, y_bits):
    x_binary = binary[:x_bits]
    y_binary = binary[x_bits:]
    x_decimal = int(x_binary, 2)
    y_decimal = int(y_binary, 2)
    x = x_bounds[0] + (x_bounds[1] - x_bounds[0]) * (x_decimal / (2 ** x_bits - 1))
    y = y_bounds[0] + (y_bounds[1] - y_bounds[0]) * (y_decimal / (2 ** y_bits - 1))
    return x, y

def fitness_function(individual, x_bounds, y_bounds, x_bits, y_bits):
    x, y = binary_to_float(individual, x_bounds, y_bounds, x_bits, y_bits)
    return object_function(x, y)

def adjust_fitness(fitnesses):
    max_fitness = max(fitnesses)
    adjusted_fitnesses = [max_fitness - fitness for fitness in fitnesses]
    return adjusted_fitnesses

def roulette_wheel_selection(population, fitnesses):
    adjusted_fitnesses = adjust_fitness(fitnesses)
    total_fitness = sum(adjusted_fitnesses)
    pick = random.uniform(0, total_fitness)
    current = 0
    for individual, fitness in zip(population, adjusted_fitnesses):
        current += fitness
        if current > pick:
            return individual
    return population[-1]

def crossover(off_1, off_2, crossover_rate=0.8):
    if random.random() < crossover_rate:
        point = random.randint(1, len(off_1) - 1)
        new_off_1 = off_1[:point] + off_2[point:]
        new_off_2 = off_2[:point] + off_1[point:]
        return new_off_1, new_off_2
    return off_1, off_2

def mutate(chrom, mutation_rate=0.05):
    chrom_list = list(chrom)
    for index in range(len(chrom_list)):
        if random.random() < mutation_rate:
            chrom_list[index] = '0' if chrom_list[index] == '1' else '1'
    return ''.join(chrom_list)

def genetic_algorithm(x_bounds, y_bounds, x_bits, y_bits, population_size=100, generations=100):
    population = [''.join([str(np.random.randint(2)) for _ in range(x_bits + y_bits)]) for _ in range(population_size)]

    for _ in range(generations):
        fitnesses = [fitness_function(individual, x_bounds, y_bounds, x_bits, y_bits) for individual in population]

        new_population = []
        for _ in range(population_size // 2):
            parent1 = roulette_wheel_selection(population, fitnesses)
            parent2 = roulette_wheel_selection(population, fitnesses)

            offspring1, offspring2 = crossover(parent1, parent2)
            offspring1 = mutate(offspring1)
            offspring2 = mutate(offspring2)

            new_population.extend([offspring1, offspring2])

        population = new_population

    best_fitnesses = [fitness_function(individual, x_bounds, y_bounds, x_bits, y_bits) for individual in population]
    best_index = best_fitnesses.index(min(best_fitnesses))
    best_x, best_y = binary_to_float(population[best_index], x_bounds, y_bounds, x_bits, y_bits)
    best_fitness = best_fitnesses[best_index]

    print(f"Best solution: x={best_x}, y={best_y}, Best fitness: {best_fitness}")

if __name__ == "__main__":
    # 소수점 5번째까지 표현
    digit = 5

    # x, y의 제약 범위
    x_bounds = [-3.0, 3.0]
    y_bounds = [-2.0, 2.0]

    # 각 제약식의 이진수 비트 변환 자릿수 계산
    x_range = (x_bounds[-1]-x_bounds[0])*(10**(digit-1))
    y_range = (y_bounds[-1]-y_bounds[0])*(10**(digit-1))

    x_bits = math.ceil(math.log2(x_range))
    y_bits = math.ceil(math.log2(y_range))
    genetic_algorithm(x_bounds, y_bounds, x_bits, y_bits)