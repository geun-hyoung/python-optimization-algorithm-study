import math
import time

# 목적 함수
def objective_function(x, y):
    return 21.5 + x * math.sin(4 * math.pi * x) + y * math.sin(20 * math.pi * y)

# 실수 범위 내에서 임의의 값을 생성
def random_step(start, bounds, step_size):
    # step_size를 기반으로 새로운 값 생성
    return max(min(start + uniform(-step_size, step_size), bounds[1]), bounds[0])

# Hill-Climbing 알고리즘 적용
def hill_climbing_real(start, bounds_x, bounds_y, iterations, step_size):
    start_time = time.time()

    best_x, best_y = start
    best_value = objective_function(best_x, best_y)

    for _ in range(iterations):
        candidate_x = random_step(best_x, bounds_x, step_size)
        candidate_y = random_step(best_y, bounds_y, step_size)
        candidate_value = objective_function(candidate_x, candidate_y)

        if candidate_value > best_value:
            best_x, best_y, best_value = candidate_x, candidate_y, candidate_value

    end_time = time.time()
    cpu_time = end_time - start_time  # 소요 시간 계산
    return best_x, best_y, best_value, cpu_time

if __name__ == "__main__":
    # x, y의 제약 범위
    x_bounds = [-3.0, 12.1]
    y_bounds = [4.1, 5.8]

    iterations_list = [500, 1000, 1500]
    step_size = 0.01  # 임의로 설정한 step_size, 필요에 따라 조정 가능

    # 초기해
    init_x = uniform(x_bounds[0], x_bounds[1])
    init_y = uniform(y_bounds[0], y_bounds[1])

    for iterations in iterations_list:
        best_x, best_y, best_value, cpu_time = hill_climbing_real((init_x, init_y), x_bounds, y_bounds, iterations, step_size)
        print(f"반복 횟수: {iterations}, 최적의 x: {best_x:.5f}, 최적의 y: {best_y:.5f}, 최대값: {best_value:.5f}, CPU 시간: {cpu_time:.7f}초")