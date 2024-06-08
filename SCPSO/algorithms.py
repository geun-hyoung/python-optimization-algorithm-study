import numpy as np
import random
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans

class Particle:
    def __init__(self, dimension, num_clusters):
        self.position = np.random.rand(num_clusters, dimension)  # 각 군집 중심점의 위치
        self.velocity = np.random.rand(num_clusters, dimension)  # 속도 초기화
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')

# Fitness function 수정: 클러스터 내 거리의 합 최소화
def fitness_function(cluster_centers, Y):
    distances = np.sqrt(((Y[:, np.newaxis, :] - cluster_centers) ** 2).sum(axis=2))
    closest = np.argmin(distances, axis=1)
    inertia = sum([np.min(distances[i]) for i in range(distances.shape[0])])
    return inertia

def pso(Y, C, num_particles, max_iter, inertia_weight, c1, c2):
    num_features = Y.shape[1]
    particles = [Particle(num_features, C) for _ in range(num_particles)]
    global_best_position = None
    global_best_fitness = float('inf')

    for _ in range(max_iter):
        for particle in particles:
            fitness = fitness_function(particle.position, Y)
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle.position.copy()

        for particle in particles:
            r1 = np.random.rand(C, num_features)
            r2 = np.random.rand(C, num_features)

            cognitive_component = c1 * r1 * (particle.best_position - particle.position)
            social_component = c2 * r2 * (global_best_position - particle.position)
            particle.velocity = inertia_weight * particle.velocity + cognitive_component + social_component
            particle.position += particle.velocity

    km = KMeans(n_clusters=C, init=global_best_position, n_init=1)
    km.fit(Y)
    return km.labels_

def initialize_clusters(num_clusters, data):
    num_samples = data.shape[0]
    indices = random.sample(range(num_samples), num_clusters)
    return data[indices]

def calculate_intra_cluster_distance(cluster_centers, data):
    distances = euclidean_distances(data, cluster_centers)
    closest = np.argmin(distances, axis=1)
    total_distance = sum([np.min(distances[i]) for i in range(distances.shape[0])])
    return total_distance

def selection(population, fitness_scores):
    selected_indices = np.argsort(fitness_scores)[:len(population) // 2]
    return [population[i] for i in selected_indices]

def crossover(parent1, parent2):
    crossover_point = random.randint(0, parent1.shape[0] - 1)
    child1 = np.vstack((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.vstack((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

def mutation(cluster_centers, data):
    mutation_point = random.randint(0, cluster_centers.shape[0] - 1)
    cluster_centers[mutation_point] = data[random.randint(0, data.shape[0] - 1)]
    return cluster_centers

# 유전 알고리즘 메인 함수
def ga(data, num_clusters, num_generations, population_size):
    population = [initialize_clusters(num_clusters, data) for _ in range(population_size)]
    best_solution = None
    best_fitness = float('inf')

    for generation in range(num_generations):
        fitness_scores = [calculate_intra_cluster_distance(clusters, data) for clusters in population]

        if min(fitness_scores) < best_fitness:
            best_fitness = min(fitness_scores)
            best_solution = population[np.argmin(fitness_scores)]

        selected_population = selection(population, fitness_scores)

        next_population = []
        while len(next_population) < population_size:
            parent1, parent2 = random.sample(selected_population, 2)
            child1, child2 = crossover(parent1, parent2)
            next_population.append(mutation(child1, data))
            if len(next_population) < population_size:
                next_population.append(mutation(child2, data))

        population = next_population

    km = KMeans(n_clusters=num_clusters, init=best_solution, n_init=1)
    km.fit(data)
    return km.labels_