import numpy as np
import random

# 초기 pheromone_matrix, 이 때 n*n 행렬을 생성함
def initialize_pheromone_matrix(n, initial_pheromone):
    return np.full((n, n), initial_pheromone)

def calculate_probabilities(distances, pheromones, alpha, beta):
    heuristic_information = np.where(distances > 0, 1 / distances, 0)

    probabilities = (pheromones ** alpha) * (heuristic_information ** beta)
    probabilities_sum = probabilities.sum(axis=1, keepdims=True)

    probabilities_sum[probabilities_sum == 0] = 1
    probabilities = probabilities / probabilities_sum

    return probabilities

# 확률을 기반으로 다음 경로 설정, 초기 도시는 랜덤
def construct_solution(distances, probabilities):
    n = distances.shape[0]
    path = [np.random.randint(n)]
    for _ in range(n - 1):
        current_city = path[-1]
        next_city_prob = probabilities[current_city].copy()
        next_city_prob[path] = 0
        next_city_prob_sum = next_city_prob.sum()

        if next_city_prob_sum == 0:
            next_city_prob = np.ones_like(next_city_prob) / (n - len(path))
        else:
            next_city_prob /= next_city_prob_sum

        next_city_prob = next_city_prob / next_city_prob.sum()  # 확률이 1로 합산되도록 보장
        next_city = np.random.choice(np.arange(n), p=next_city_prob)
        path.append(next_city)
    return path

# 페로몬 업데이트, [1]에는 cost가 담겨있음, 전체 최적의 해를 찾고 페로몬 업데이트
def update_pheromones(pheromones, paths, n_best, decay):
    sorted_paths = sorted(paths, key=lambda x: x[1])
    pheromones *= decay

    for path, cost in sorted_paths[:n_best]:
        pheromone_deposit = 1.0 / cost  # 비용의 역수를 페로몬 분비량으로 사용
        for i in range(len(path) - 1):
            pheromones[path[i], path[i + 1]] += pheromone_deposit
        pheromones[path[-1], path[0]] += pheromone_deposit  # 마지막 도시에서 첫 도시로 돌아가는 경로 업데이트
    return pheromones

def calculate_cost(path, distances):
    return sum(distances[path[i], path[i + 1]] for i in range(len(path) - 1)) + distances[path[-1], path[0]]

def ant_colony_optimization(distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=2, initial_pheromone=1.0):
    n = distances.shape[0]
    pheromones = initialize_pheromone_matrix(n, initial_pheromone)
    best_path = None
    best_cost = 9999
    for _ in range(n_iterations):
        paths = []
        probabilities = calculate_probabilities(distances, pheromones, alpha, beta)
        for _ in range(n_ants):
            path = construct_solution(distances, probabilities)
            cost = calculate_cost(path, distances)
            paths.append((path, cost))
            if cost < best_cost:
                best_path, best_cost = path, cost
        pheromones = update_pheromones(pheromones, paths, n_best, decay)
    return best_path, best_cost

if __name__ == "__main__":
    matrix = [
        [9999, 3, 5, 48, 48, 8, 8, 5, 5, 3, 3, 0, 3, 5, 8, 8, 5],
        [3, 9999, 3, 48, 48, 8, 8, 5, 5, 0, 0, 3, 0, 3, 8, 8, 5],
        [5, 3, 9999, 72, 72, 48, 48, 24, 24, 3, 3, 5, 3, 0, 48, 48, 24],
        [48, 48, 74, 9999, 0, 6, 6, 12, 12, 48, 48, 48, 48, 74, 6, 6, 12],
        [48, 48, 74, 0, 9999, 6, 6, 12, 12, 48, 48, 48, 48, 74, 6, 6, 12],
        [8, 8, 50, 6, 6, 9999, 0, 8, 8, 8, 8, 8, 8, 50, 0, 0, 8],
        [8, 8, 50, 6, 6, 0, 9999, 8, 8, 8, 8, 8, 8, 50, 0, 0, 8],
        [5, 5, 26, 12, 12, 8, 8, 9999, 0, 5, 5, 5, 5, 26, 8, 8, 0],
        [5, 5, 26, 12, 12, 8, 8, 0, 9999, 5, 5, 5, 5, 26, 8, 8, 0],
        [3, 0, 3, 48, 48, 8, 8, 5, 5, 9999, 0, 3, 0, 3, 8, 8, 5],
        [3, 0, 3, 48, 48, 8, 8, 5, 5, 0, 9999, 3, 0, 3, 8, 8, 5],
        [0, 3, 5, 48, 48, 8, 8, 5, 5, 3, 3, 9999, 3, 5, 8, 8, 5],
        [3, 0, 3, 48, 48, 8, 8, 5, 5, 0, 0, 3, 9999, 3, 8, 8, 5],
        [5, 3, 0, 72, 72, 48, 48, 24, 24, 3, 3, 5, 3, 9999, 48, 48, 24],
        [8, 8, 50, 6, 6, 0, 0, 8, 8, 8, 8, 8, 8, 50, 9999, 0, 8],
        [8, 8, 50, 6, 6, 0, 0, 8, 8, 8, 8, 8, 8, 50, 0, 9999, 8],
        [5, 5, 26, 12, 12, 8, 8, 0, 0, 5, 5, 5, 5, 26, 8, 8, 9999]
    ]

    distances = np.array(matrix)
    random.seed(2)
    n_ants = 50
    n_best = 4
    n_iterations = 1000
    decay = 0.95
    alpha = 1
    beta = 2

    best_path, best_cost = ant_colony_optimization(distances, n_ants, n_best, n_iterations, decay, alpha, beta)
    print("Best path:", best_path)
    print("Best cost:", best_cost)

