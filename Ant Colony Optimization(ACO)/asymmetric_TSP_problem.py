import numpy as np
import random

# 초기 pheromone_matrix, 이 때 n*n 행렬을 생성함
def initialize_pheromone_matrix(n, initial_pheromone):
    return np.full((n, n), initial_pheromone)

def calculate_probabilities(distances, pheromones, alpha, beta):

    probabilities_sum = probabilities.sum(axis=1, keepdims=True)


    return probabilities

# 확률을 기반으로 다음 경로 설정, 초기 도시는 랜덤
    n = distances.shape[0]
    path = [np.random.randint(n)]
        current_city = path[-1]

        else:

        path.append(next_city)
    return path

# 페로몬 업데이트, [1]에는 cost가 담겨있음, 전체 최적의 해를 찾고 페로몬 업데이트

    return pheromones

def calculate_cost(path, distances):
    return sum(distances[path[i], path[i + 1]] for i in range(len(path) - 1)) + distances[path[-1], path[0]]

    n = distances.shape[0]
    pheromones = initialize_pheromone_matrix(n, initial_pheromone)
    best_path = None
    best_cost = 9999
    for _ in range(n_iterations):
        paths = []
        probabilities = calculate_probabilities(distances, pheromones, alpha, beta)
        for _ in range(n_ants):
            cost = calculate_cost(path, distances)
            paths.append((path, cost))
            if cost < best_cost:
                best_path, best_cost = path, cost
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
    n_iterations = 1000
    beta = 2

    print("Best path:", best_path)
    print("Best cost:", best_cost)