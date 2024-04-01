import numpy as np
from scipy.linalg import inv
import random
import time

# Rosenbrock 함수
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def is_ideal_solution(x, y, epsilon=0.01):
    return abs(x - 1) < epsilon and abs(y - 1) < epsilon

# 그래디언트
def gradient(x, y):
    return np.array([-2*(1 - x) - 400*x*(y - x**2), 200*(y - x**2)])

# 헤시안
def hessian(x, y):
    return np.array([[2 - 400*y + 1200*x**2, -400*x], [-400*x, 200]])

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

def newtons_method(start_x, start_y, num_iterations):
    x, y = start_x, start_y

    for i in range(num_iterations):
        if is_ideal_solution(x, y):
            return x, y, rosenbrock(x, y), i + 1

        grad = np.array([-2 * (1 - x) - 400 * x * (y - x**2), 200 * (y - x**2)])
        hessian = np.array([[1200 * x**2 - 400 * y + 2, -400 * x], [-400 * x, 200]])
        hessian_inv = np.linalg.pinv(hessian)  # 유사 역행렬 사용
        update = np.matmul(hessian_inv, grad)  # 안정적인 행렬 곱셈 사용

        x, y = x - update[0], y - update[1]
        x, y = np.nan_to_num(x), np.nan_to_num(y)  # NaN 또는 inf를 0으로 대체

    return x, y, rosenbrock(x, y), num_iterations

# 힐 클라이밍
def hill_climbing(start_x, start_y, num_iterations, step_size=0.001):
    x, y = start_x, start_y

    for i in range(num_iterations):
        if is_ideal_solution(x, y):
            return x, y, rosenbrock(x, y), i + 1

        next_x = x + random.uniform(-step_size, step_size)
        next_y = y + random.uniform(-step_size, step_size)

        if rosenbrock(next_x, next_y) < rosenbrock(x, y):
            x, y = next_x, next_y

    return x, y, rosenbrock(x, y), num_iterations

if __name__ == "__main__":
    start_x, start_y = 2, 0  # 시작 위치
    alpha = 0.001  # 학습률 = alpha
    num_iterations = 10000  # 반복 횟수

    start_time = time.time()
    gd_result  = gradient_descent(start_x, start_y, alpha, num_iterations)
    gd_time = time.time() - start_time

    start_time = time.time()
    nm_result = newtons_method(start_x, start_y, num_iterations)
    nm_time  = time.time() - start_time

    start_time = time.time()
    hc_result  = hill_climbing(start_x, start_y, num_iterations)
    hc_time = time.time() - start_time

    # 결과 출력
    print("Gradient Descent: 결과값 = {}, 시간 = {:.7f}초".format(gd_result, gd_time))
    print("Newton's Method: 결과값 = {}, 시간 = {:.7f}초".format(nm_result, nm_time))
    print("Hill Climbing: 결과값 = {}, 시간 = {:.7f}초".format(hc_result, hc_time))