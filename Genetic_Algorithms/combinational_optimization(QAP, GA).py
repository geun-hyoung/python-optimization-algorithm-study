import numpy as np
import random

# 행렬 분할
def extract_distance_and_flow(matrix):
    distance = np.zeros((15, 15))
    flow = np.zeros((15, 15))

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i < j:  # 대각선 상반부
                distance[i, j] = matrix[i, j]
            elif i > j:  # 대각선 하반부
                flow[i, j] = matrix[i, j]

    # 대칭 채우기
    distance += distance.T
    flow += flow.T
    return distance, flow

# 목적 함수
def objective_function(solution, distance, flow):
    cost = 0
    for i in range(len(solution)):
        for j in range(len(solution)):
            cost += flow[i][j] * distance[solution[i]][solution[j]]
    return cost

def create_initial_population(pop_size, solution_size):
    return [random.sample(range(solution_size), solution_size) for _ in range(pop_size)]

def evaluate_population(population, objective_function, distance, flow):
    return [objective_function(individual, distance, flow) for individual in population]

def select_population(population, scores, k=3):
    selected = []
    population_size = len(population)

    for _ in range(population_size):
        tournament_indices = np.random.choice(population_size, k)
        tournament_scores = [scores[i] for i in tournament_indices]

        best_index = tournament_indices[np.argmin(tournament_scores)]
        selected.append(population[best_index])

    return selected

def crossover(parent1, parent2, solution_size, crossover_rate=0.9):
    if random.random() < crossover_rate:
        child1, child2 = parent1.copy(), parent2.copy()
        start, stop = sorted(random.sample(range(solution_size), 2))

        mid_section1 = parent1[start:stop]
        mid_section2 = parent2[start:stop]

        child1 = [item for item in child1 if item not in mid_section2]
        child2 = [item for item in child2 if item not in mid_section1]

        child1[start:start] = mid_section2
        child2[start:start] = mid_section1

        return child1, child2
    else:
        return parent1, parent2

def mutate(solution, solution_size, mutation_rate=0.01):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(solution_size), 2)
        solution[idx1], solution[idx2] = solution[idx2], solution[idx1]
    return solution

def genetic_algorithm(pop_size, solution_size, crossover_rate, mutation_rate, generations, distance, flow):
    population = create_initial_population(pop_size, solution_size)
    best, best_eval = population[0], objective_function(population[0], distance, flow)

    for gen in range(generations):
        fitness = evaluate_population(population, objective_function, distance, flow)
        selected = select_population(population, fitness)
        offspring = []

        for i in range(0, len(population), 2):
            parent1, parent2 = selected[i], selected[i + 1]
            child1, child2 = crossover(parent1, parent2, solution_size, crossover_rate)
            offspring.append(mutate(child1, solution_size, mutation_rate))
            offspring.append(mutate(child2, solution_size, mutation_rate))

        population = offspring
        for i in range(len(population)):
            if objective_function(population[i], distance, flow) < best_eval:
                best, best_eval = population[i], objective_function(population[i], distance, flow)

    return best, best_eval

if __name__ == "__main__":
    pop_size = 100
    solution_size = 15
    crossover_rate = 0.9
    mutation_rate = 0.01
    generations = 100

    # 대각선을 기준으로 하반부는 흐름, 상반부는 거리
    raw_data = np.array([
        [0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6],
        [10, 0, 1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 3, 4, 5],
        [0, 1, 0, 1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 3, 4],
        [5, 3, 10, 0, 1, 4, 3, 2, 1, 2, 5, 4, 3, 2, 3],
        [1, 2, 2, 1, 0, 5, 4, 3, 2, 1, 6, 5, 4, 3, 2],
        [0, 2, 0, 1, 3, 0, 1, 2, 3, 4, 1, 2, 3, 4, 5],
        [1, 2, 2, 5, 5, 2, 0, 1, 2, 3, 2, 1, 2, 3, 4],
        [2, 3, 5, 0, 5, 2, 6, 0, 1, 2, 3, 2, 1, 2, 3],
        [2, 2, 4, 0, 5, 1, 0, 5, 0, 1, 4, 3, 2, 1, 2],
        [2, 0, 5, 2, 1, 5, 1, 2, 0, 0, 5, 4, 3, 2, 1],
        [2, 2, 2, 1, 0, 0, 5, 10, 10, 0, 0, 1, 2, 3, 4],
        [0, 0, 2, 0, 3, 0, 5, 0, 5, 4, 5, 0, 1, 2, 3],
        [4, 10, 5, 2, 0, 2, 5, 5, 10, 0, 0, 3, 0, 1, 2],
        [0, 5, 5, 5, 5, 5, 1, 0, 0, 0, 5, 3, 10, 0, 1],
        [0, 0, 5, 0, 5, 10, 0, 0, 2, 5, 0, 0, 2, 4, 0]
    ])

    distance, flow = extract_distance_and_flow(raw_data)
    best_solution, best_cost = genetic_algorithm(pop_size, solution_size, crossover_rate, mutation_rate, generations, distance, flow)
    print(f"최종 배치: {best_solution}, 비용: {best_cost / 2}")