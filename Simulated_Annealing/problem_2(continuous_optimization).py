import math
import random
import numpy as np

# Six Hump Camelback 함수 정의
def object_function(x, y):
    return (4 - 2.1*x**2 + (x**4)/3) * x**2 + x*y + (-4 + 4*y**2) * y**2

# 이웃 생성
def get_neighbor(solution, step_size=0.1):
    neighbor = solution + np.random.uniform(-step_size, step_size, 2)
    # 제약 조건에 의한 범위 제한
    neighbor[0] = max(min(neighbor[0], 3), -3)
    neighbor[1] = max(min(neighbor[1], 2), -2)
    return neighbor

# 담금질 알고리즘
def simulated_annealing(initial_solution, temp=10000, cooling_rate=0.99, stopping_temp=1, max_iterations = 1000):
    current_solution = initial_solution
    current_cost = object_function(current_solution[0], current_solution[1])
    iterations = 1

    while temp > stopping_temp or max_iterations > iterations:
        neighbor = get_neighbor(current_solution)
        neighbor_cost = object_function(neighbor[0], neighbor[1])

        if neighbor_cost < current_cost:
            current_solution = neighbor
            current_cost = neighbor_cost
        temp *= cooling_rate
        iterations += 1
    return current_solution, current_cost

# 1번 annealing schedule 2번 변경
stopping_temp = 1
init_solution = np.array([random.uniform(-3, 3), random.uniform(-2, 2)])

candidate = [
    [100, 0.99],
    [1000, 0.8],
]

for obj in candidate:
    temp, cooling_rate = obj[0], obj[1]
    best_solution, best_cost = simulated_annealing(init_solution, temp, cooling_rate, stopping_temp)
    print(f"temp: {temp}, cooling_rate = {cooling_rate}, Best solution: {best_solution}, Best cost: {best_cost}")

# 2번 정지 조건 2번 바꾸기
print('\n')
temp = 100
cooling_rate = 0.99
init_solution = np.array([random.uniform(-3, 3), random.uniform(-2, 2)])
stopping_temp = 1

best_solution, best_cost = simulated_annealing(init_solution, temp, cooling_rate, stopping_temp)
print(f"stop_condition: {stopping_temp}, iterations = {1000}, Best solution: {best_solution}, Best cost: {best_cost}")

iterations = 100
best_solution, best_cost = simulated_annealing(init_solution, temp, cooling_rate, stopping_temp, iterations)
print(f"stop_condition: {stopping_temp}, iterations = {iterations}, Best solution: {best_solution}, Best cost: {best_cost}")

# 3번 초기해 5번 바꾸면서 실험
print('\n')
temp = 100
cooling_rate = 0.99
stopping_temp = 1

for ord in range(5):
    init_solution = np.array([random.uniform(-3, 3), random.uniform(-2, 2)])
    best_solution, best_cost = simulated_annealing(init_solution, temp, cooling_rate, stopping_temp)
    print(f"change_order: {ord + 1}, Best solution: {best_solution}, Best cost: {best_cost}")

# 4번 랜덤 시드 10번
print('\n')
temp = 100
cooling_rate = 0.99
init_solution = np.array([random.uniform(-3, 3), random.uniform(-2, 2)])
stopping_temp = 1

for seed in range(10):
    random.seed(seed)
    # SA 실행
    best_solution, best_cost = simulated_annealing(init_solution, temp, cooling_rate, stopping_temp)
    print(f"Seed: {seed}, Best solution: {best_solution}, Best cost: {best_cost}")

print("Problem 1 - SA Algorithms done")