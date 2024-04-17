import numpy as np
import random
import math

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

# 목적 함수(양방향 흐름 모두 고려)
def objective_function(solution):
    cost = 0
    for i in range(len(solution)):
        for j in range(len(solution)):
            # 물리적 거리는 고정
            cost += flow[i][j] * distance[solution[i]][solution[j]]
    return cost

# 이웃 정의 - 3D > 1D로 생각해서 두 자리를 변환
def get_neighbor(solution):
    neighbor = solution.copy()
    i, j = random.sample(range(len(solution)), 2)
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor

# SA 알고리즘
def simulated_annealing(temp, init_solution, cooling_rate, stop_condition, max_iterations = 1000):
    current_solution = init_solution
    current_cost = objective_function(current_solution)
    iterations = 1

    while temp > stop_condition or max_iterations > iterations:
        new_solution = get_neighbor(current_solution)
        new_cost = objective_function(new_solution)
        cost_difference = new_cost - current_cost

        # Improving move + Non-improving move
        if cost_difference < 0 or math.exp(-cost_difference / temp) > random.random():
            current_solution = new_solution
            current_cost = new_cost

        temp *= cooling_rate
        iterations += 1

    return current_solution, current_cost

# 거리, 흐름 구하기
distance = np.zeros((15, 15))
flow = np.zeros((15, 15))

for i in range(raw_data.shape[0]):
    for j in range(raw_data.shape[1]):
        if i < j:  # 대각선 상반부
            distance[i, j] = raw_data[i, j]
        elif i > j:  # 대각선 하반부
            flow[i, j] = raw_data[i, j]

# 대칭 채우기
distance += distance.T
flow += flow.T

# 초기 파라미터 설정
temp = 10000
init_solution = random.sample(range(15), 15)
cooling_rate = 0.99
stop_condition = 1
iterations = 1000

# 1번 annealing schedule 2번 변경
candidate = [[10000, 0.99], [10000, 0.8]]

for obj in candidate:
    ch1_temp, ch1_cooling_rate = obj[0], obj[1]
    best_solution, best_cost = simulated_annealing(ch1_temp, init_solution, ch1_cooling_rate, stop_condition, iterations)
    print(f"temp: {ch1_temp}, cooling_rate = {ch1_cooling_rate}, Best solution: {best_solution}, Best cost: {best_cost/2}")

# 2번 정지 조건 2번 바꾸기
print('\n')
ch2_stop_condition = 1

best_solution, best_cost = simulated_annealing(temp, init_solution, cooling_rate, stop_condition, iterations)
print(f"stop_condition: {ch2_stop_condition}, iterations = {iterations}, Best solution: {best_solution}, Best cost: {best_cost/2}")

ch2_stop_condition = 0.5
best_solution, best_cost = simulated_annealing(temp, init_solution, cooling_rate, stop_condition, iterations)
print(f"stop_condition: {ch2_stop_condition}, iterations = {iterations}, Best solution: {best_solution}, Best cost: {best_cost/2}")

# 3번 초기해 5번 바꾸면서 실험
print('\n')
for ord in range(5):
    ch3_init_solution = random.sample(range(15), 15)
    best_solution, best_cost = simulated_annealing(temp, ch3_init_solution, cooling_rate, stop_condition, iterations)
    print(f"change_order: {ord + 1}, Best solution: {best_solution}, Best cost: {best_cost/2}")

# 4번 랜덤 시드 10번
print('\n')
for seed in range(10):
    random.seed(seed)
    # SA 실행
    best_solution, best_cost = simulated_annealing(temp, init_solution, cooling_rate, stop_condition, iterations)
    print(f"Seed: {seed+1}, Best solution: {best_solution}, Best cost: {best_cost/2}")

print("\nProblem 1 - SA Algorithms done")