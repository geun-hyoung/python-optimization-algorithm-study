import numpy as np
import random

def objective_function(solution, distance, flow):
    cost = 0
    for i in range(len(solution)):
        for j in range(len(solution)):
            cost += flow[i, j] * distance[solution[i], solution[j]]
    return cost

def extract_distance_and_flow(matrix):
    distance = np.zeros((15, 15))
    flow = np.zeros((15, 15))

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i < j:
                distance[i, j] = matrix[i, j]
            elif i > j:
                flow[i, j] = matrix[i, j]

    distance += distance.T
    flow += flow.T
    return distance, flow

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def update_velocity(velocity, personal_best_position, global_best_position, position, w=0.5, c1=0.8, c2=0.9):
    r1, r2 = random.random(), random.random()
    component_1 = c1 * r1 * (personal_best_position - position)    # PB
    component_2 = c2 * r2 * (global_best_position - position)   # GB
    new_velocity = w * velocity + component_1 + component_2
    return new_velocity

def update_position(position, velocity):
    new_position = position.copy()
    n = len(velocity)

    probabilities = softmax(velocity)

    for i in range(n):
        if random.random() < probabilities[i]:
            swap_idx = (i + random.randint(1, n - 1)) % n
            new_position[i], new_position[swap_idx] = new_position[swap_idx], new_position[i]

    return new_position

def pso_for_qap(distance, flow, particle_num=30, iterations=100):
    num_dimensions = len(distance)
    particles = [np.random.permutation(num_dimensions) for _ in range(particle_num)]
    velocities = [np.zeros(num_dimensions) for _ in range(particle_num)]

    personal_best_positions = particles.copy()
    personal_best_scores = [float('inf')] * particle_num

    global_best_score = float('inf')
    global_best_position = None

    for _ in range(iterations):
        for i in range(particle_num):
            cost = objective_function(particles[i], distance, flow)

            if cost < personal_best_scores[i]:
                personal_best_scores[i] = cost
                personal_best_positions[i] = particles[i]

                if cost < global_best_score:
                    global_best_score = cost
                    global_best_position = particles[i]

        for i in range(particle_num):
            velocities[i] = update_velocity(velocities[i], personal_best_positions[i], global_best_position, particles[i])
            particles[i] = update_position(particles[i], velocities[i])

    return global_best_position, global_best_score

if __name__ == "__main__":
    particle_num = 30
    iterations = 1000

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

    distance, flow = extract_distance_and_flow(raw_data)
    best_solution, best_cost = pso_for_qap(distance, flow, particle_num, iterations)
    print(f"최종 배치: {best_solution}, 비용: {best_cost/2}")