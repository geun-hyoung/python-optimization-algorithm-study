import time

from gradient_ascent_search import gradient_descent
from newton_method import newtons_method
from hill_climbing import hill_climbing

if __name__ == "__main__":
    start_x, start_y = 1, 0  # 시작 위치
    alpha = 0.001  # 학습률 = alpha
    num_iterations = 10000  # 반복 횟수

    start_time = time.time()
    gd_result  = gradient_descent(start_x, start_y, alpha, num_iterations)
    gd_time = time.time() - start_time

    start_time = time.time()
    nm_result = newtons_method(start_x, start_y, num_iterations, alpha)
    nm_time  = time.time() - start_time

    start_time = time.time()
    hc_result  = hill_climbing(start_x, start_y, num_iterations, alpha)
    hc_time = time.time() - start_time

    # 결과 출력
    print("Gradient Descent: 결과값 = {}, 시간 = {:.7f}초".format(gd_result, gd_time))
    print("Newton's Method: 결과값 = {}, 시간 = {:.7f}초".format(nm_result, nm_time))
    print("Hill Climbing: 결과값 = {}, 시간 = {:.7f}초".format(hc_result, hc_time))