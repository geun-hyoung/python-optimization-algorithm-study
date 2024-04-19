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

# 2차 미분 : 헤시안
def hessian(x, y):
    return np.array([[2 - 400*y + 1200*x**2, -400*x], [-400*x, 200]])

def newtons_method(start_x, start_y, num_iterations, alpha):
    x, y = start_x, start_y

    for i in range(num_iterations):
        if is_ideal_solution(x, y):
            return x, y, rosenbrock(x, y), i + 1

        grad = gradient(x, y)
        hessian_matrix = hessian(x, y)
        hessian_inv = np.linalg.pinv(hessian_matrix)  # 역행렬
        update = np.matmul(hessian_inv, grad)

        x, y = x - alpha*update[0], y - alpha*update[1]
        x, y = np.nan_to_num(x), np.nan_to_num(y)

    return x, y, rosenbrock(x, y), num_iterations
