import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh


# Step 1: 샘플 문서 선택
def select_sample_documents(D, n1):
    indices = np.random.choice(len(D), n1, replace=False)
    return D[indices], indices


# Step 2: 유사도 그래프 정의
def define_similarity_graph(S):
    return cosine_similarity(S)


# Step 3: 유사도 행렬 추정
def estimate_similarity_matrix(S1, D, indices):
    S2 = np.zeros((len(S1), len(D)))
    for i in range(len(S1)):
        for j in range(len(D)):
            if j not in indices:
                S2[i, j] = cosine_similarity(S1[i].reshape(1, -1), D[j].reshape(1, -1))[0][0]
    return S2


# Step 4: 대각 행렬 정의
def define_diagonal_matrix(S):
    return np.diag(np.sum(S, axis=1))


# Step 5: 라플라시안 행렬 구성
def construct_laplacian_matrix(D, W):
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    return D_inv_sqrt @ W @ D_inv_sqrt


# Step 6: 가장 큰 고유 벡터 찾기
def find_largest_eigen_vectors(LM, k):
    _, eig_vectors = eigsh(LM, k=k, which='LM')
    return eig_vectors


# Step 7: 행렬 Y 구성
def construct_matrix_Y(V):
    return V / np.linalg.norm(V, axis=1, keepdims=True)

class Particle:
    def __init__(self, dimension):
        self.position = np.random.rand(dimension)
        self.velocity = np.random.rand(dimension)
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')

def fitness_function(particle_position, Y, C):
    km = KMeans(n_clusters=C)
    weighted_Y = Y * particle_position
    km.fit(weighted_Y)
    return km.inertia_

def pso(Y, C, num_particles, max_iter, inertia, c1, c2):
    dimension = Y.shape[1]
    particles = [Particle(dimension) for _ in range(num_particles)]
    global_best_position = particles[0].position.copy()
    global_best_fitness = float('inf')

    for _ in range(max_iter):
        for particle in particles:
            fitness = fitness_function(particle.position, Y, C)
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle.position.copy()

        for particle in particles:
            r1 = np.random.rand(dimension)
            r2 = np.random.rand(dimension)

            cognitive_component = c1 * r1 * (particle.best_position - particle.position)
            social_component = c2 * r2 * (global_best_position - particle.position)
            particle.velocity = inertia * particle.velocity + cognitive_component + social_component
            particle.position += particle.velocity

    return global_best_position

# 전체 과정 수행
def spectral_clustering_with_pso(D, k, num_particles, max_iter, inertia, c1, c2):
    W = define_similarity_graph(D)
    D_matrix = define_diagonal_matrix(W)
    LM = construct_laplacian_matrix(D_matrix, W)
    V = find_largest_eigen_vectors(LM, k)
    Y = construct_matrix_Y(V)

    best_position = pso(Y, k, num_particles, max_iter, inertia, c1, c2)

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(Y * best_position)
    return kmeans.labels_