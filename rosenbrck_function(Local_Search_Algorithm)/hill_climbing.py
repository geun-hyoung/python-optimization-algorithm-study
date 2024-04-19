import numpy as np
import random
import time

# Rosenbrock : 목적 함수
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

# ideal solution의 범위 생성
def is_ideal_solution(x, y, epsilon=0.01):
    return abs(x - 1) < epsilon and abs(y - 1) < epsilon

# 힐 클라이밍
def hill_climbing(start_x, start_y, num_iterations, step_size):
    x, y = start_x, start_y

    for i in range(num_iterations):
        if is_ideal_solution(x, y):
            return x, y, rosenbrock(x, y), i + 1

        next_x = x + random.uniform(-step_size, step_size)
        next_y = y + random.uniform(-step_size, step_size)

        if rosenbrock(next_x, next_y) < rosenbrock(x, y):
            x, y = next_x, next_y

    return x, y, rosenbrock(x, y), num_iterations