import numpy as np
import random

# 초기 pheromone_matrix, 이 때 n*n 행렬을 생성함
def initialize_pheromone_matrix(n, initial_pheromone):
    return np.full((n, n), initial_pheromone)

def calculate_probabilities(distances, pheromones, alpha, beta):
    heuristic_information = np.where(distances > 0, 1 / distances, 0)  # 거리가 0인 경우 방지

    probabilities = (pheromones ** alpha) * (heuristic_information ** beta)  # 각 경로 속 페로몬의 양 * 중요도(1/거리)
    probabilities_sum = probabilities.sum(axis=1, keepdims=True)

    probabilities_sum[probabilities_sum == 0] = 1  # 합이 0인 경우 방지
    probabilities = probabilities / probabilities_sum  # 다음 노드로 가기 위해 확률 값 구하기

    return probabilities

def select_next_city(current_city, probabilities, q0):
    if np.random.rand() < q0:
        next_city = np.argmax(probabilities)
    else:
        next_city = np.random.choice(np.arange(len(probabilities)), p=probabilities)
    return next_city

# 확률을 기반으로 다음 경로 설정, 초기 도시는 랜덤
def construct_solution(distances, probabilities, q0):
    n = distances.shape[0]
    path = [np.random.randint(n)]
    visited = set(path)  # 방문한 도시를 추적하기 위한 집합
    while len(path) < n:
        current_city = path[-1]
        probabilities_copy = probabilities[current_city].copy()

        # 방문한 도시는 확률을 0으로 설정
        for city in visited:
            probabilities_copy[city] = 0

        probabilities_sum = probabilities_copy.sum()
        if probabilities_sum > 0:
            probabilities_copy /= probabilities_sum
        else:
            break

        next_city = select_next_city(current_city, probabilities_copy, q0)
        if next_city in visited:
            break  # 이미 방문한 도시인 경우 루프 종료

        visited.add(next_city)
        path.append(next_city)

    # 모든 도시를 방문하지 못한 경우, 남은 도시를 추가
    if len(path) < n:
        remaining_cities = set(range(n)) - visited
        path.extend(remaining_cities)

    return path

def local_pheromone_update(pheromones, path, local_pheromone_value, decay):
    for i in range(len(path) - 1):
        pheromones[path[i], path[i + 1]] = (1 - decay) * pheromones[path[i], path[i + 1]] + decay * 0.008
    pheromones[path[-1], path[0]] = (1 - decay) * pheromones[path[-1], path[0]] + decay * 0.008
    return pheromones

# 페로몬 업데이트, [1]에는 cost가 담겨있음, 전체 최적의 해를 찾고 페로몬 업데이트
def update_pheromones(pheromones, best_path, best_cost, decay):
    pheromone_deposit = 1.0 / best_cost  # 비용의 역수를 페로몬 분비량으로 사용

    for i in range(len(best_path) - 1):
        pheromones[best_path[i], best_path[i + 1]] *= (1-decay)
        pheromones[best_path[i], best_path[i + 1]] += pheromone_deposit
    pheromones[best_path[-1], best_path[0]] += pheromone_deposit
    return pheromones

def calculate_cost(path, distances):
    return sum(distances[path[i], path[i + 1]] for i in range(len(path) - 1)) + distances[path[-1], path[0]]

def ant_colony_optimization(distances, n_ants, n_best, n_iterations, decay, alpha, beta, initial_pheromone, q0, local_pheromone_value):
    n = distances.shape[0]
    pheromones = initialize_pheromone_matrix(n, initial_pheromone)
    best_path = None
    best_cost = 9999
    for _ in range(n_iterations):
        paths = []
        probabilities = calculate_probabilities(distances, pheromones, alpha, beta)

        for _ in range(n_ants):
            path = construct_solution(distances, probabilities, q0)
            pheromones = local_pheromone_update(pheromones, path, local_pheromone_value, decay)  # 지역 페로몬 업데이트
            cost = calculate_cost(path, distances)
            paths.append((path, cost))
            if cost < best_cost:
                best_path, best_cost = path, cost
        pheromones = update_pheromones(pheromones, best_path, best_cost, decay)
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
    n_ants = 500
    n_best = 1
    n_iterations = 1000
    decay = 0.8
    alpha = 2
    beta = 2
    initial_pheromone = 0.05
    q0 = 0.5
    local_pheromone_value = 0.02

    best_path, best_cost = ant_colony_optimization(distances, n_ants, n_best, n_iterations, decay, alpha, beta, initial_pheromone, q0, local_pheromone_value)
    print("Best path:", best_path)
    print("Best cost:", best_cost)