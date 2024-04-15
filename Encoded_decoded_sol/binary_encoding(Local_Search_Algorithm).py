import numpy as np
import math
import time
from random import sample
import matplotlib.pyplot as plt

# 목적 함수
def objective_function(x, y):
    return 21.5 + x * math.sin(4 * math.pi * x) + y * math.sin(20 * math.pi * y)

# 해당 위치의 비트 변경 : binary encoding 부분 변경
def flip_bit(binary, position):
    for pos in position:
        if binary[pos] == '0':
            binary = binary[:pos] + '1' + binary[pos + 1:]
        else:
            binary = binary[:pos] + '0' + binary[pos + 1:]
    return binary

# 이진수를 실수 값으로 디코딩
def binary_to_real_no_norm(binary, x_bounds, y_bounds, x_bits, y_bits):
    # binary 분할
    x_binary = binary[:x_bits]
    y_binary = binary[x_bits:]
    # demical(substring) 구하기
    x_decimal = int(x_binary, 2)
    y_decimal = int(y_binary, 2)
    # decision_variable 구하기
    x = x_bounds[0] + (x_bounds[1] - x_bounds[0]) * (x_decimal / (2 ** x_bits - 1))
    y = y_bounds[0] + (y_bounds[1] - y_bounds[0]) * (y_decimal / (2 ** y_bits - 1))
    return x, y

# (binary)Hill-Climbing 알고리즘 적용
def hill_climbing_no_norm(start, bounds_x, bounds_y, x_bits, y_bits, iterations, step_size):
    start_time = time.time()
    # 초기해 - 랜덤으로 설정
    best_binary = start[0]
    best_value = start[1]

    # 알고리즘 적용 부분
    for _ in range(iterations):
        # 변경할 비트 자리 선정 - step_size의 수 만큼 자리를 변경
        bit_to_flip = sample(range(0, x_bits + y_bits), step_size)
        candidate_binary = flip_bit(best_binary, bit_to_flip)
        candidate_x, candidate_y = binary_to_real_no_norm(candidate_binary, bounds_x, bounds_y, x_bits, y_bits)
        candidate_value = objective_function(candidate_x, candidate_y)

        if candidate_value > best_value:
            best_binary, best_x, best_y, best_value = candidate_binary, candidate_x, candidate_y, candidate_value

    end_time = time.time()
    cpu_time = end_time - start_time  # 소요 시간 계산
    return best_x, best_y, best_value, cpu_time

if __name__ == "__main__":
    # 소수점 5번째까지 표현
    digit = 5
    # 이진수로 변환 후 몇개를 변경할지 ~= alpha
    alpha = 10

    # x, y의 제약 범위
    x_bounds = [-3.0, 12.1]
    y_bounds = [4.1, 5.8]

    # 각 제약식의 이진수 비트 변환 자릿수 계산
    x_range = (x_bounds[-1]-x_bounds[0])*(10**(digit-1))
    y_range = (y_bounds[-1]-y_bounds[0])*(10**(digit-1))

    x_bits = math.ceil(math.log2(x_range))
    y_bits = math.ceil(math.log2(y_range))

    iterations_list = [500, 1000, 1500]
    step_size = int((x_bits + y_bits)/3)

    # 초기해
    init_binary = ''.join([str(np.random.randint(2)) for _ in range(x_bits + y_bits)])
    init_x, init_x = binary_to_real_no_norm(init_binary, x_bounds, y_bounds, x_bits, y_bits)
    init_value = objective_function(init_x, init_x)
    start = [init_binary, init_value]

    for iterations in iterations_list:
        best_x, best_y, best_value, cpu_time = hill_climbing_no_norm(start, x_bounds, y_bounds, x_bits, y_bits, iterations, step_size)
        print(f"반복 횟수: {iterations}, 최적의 x: {best_x:.5f}, 최적의 y: {best_y:.5f}, 최대값: {best_value:.5f}, CPU 시간: {cpu_time:.5f}초")
