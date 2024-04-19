import numpy as np
import random
import time

# Rosenbrock : 목적 함수
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

# ideal solution의 범위 생성
def is_ideal_solution(x, y, epsilon=0.01):
    return abs(x - 1) < epsilon and abs(y - 1) < epsilon

# 1차 미분
def gradient(x, y):
    return np.array([-2*(1 - x) - 400*x*(y - x**2), 200*(y - x**2)])

# 그래디언트 디센트
def gradient_descent(start_x, start_y, alpha, num_iterations):
    x, y = start_x, start_y

    for i in range(num_iterations):
        if is_ideal_solution(x, y):
            return x, y, rosenbrock(x, y), i + 1

        grad_x, grad_y = gradient(x, y)
        x -= alpha * grad_x
        y -= alpha * grad_y

    return x, y, rosenbrock(x, y), num_iterations