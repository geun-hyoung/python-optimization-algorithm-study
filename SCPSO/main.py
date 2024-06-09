import pandas as pd

from sklearn.preprocessing import LabelEncoder

from text_vectorization import spectral_embedding
from measure_metrics import evaluate_algorithms
from algorithms import pso, ga, SimulatedAnnealing

if __name__ == "__main__":
    # global params
    k = 15    # number_of_clusters
    max_iter = 100 # algorithms_iterations

    # pso params
    num_particles = 30
    inertia_weight = 0.5
    c1 = 1
    c2 = 1

    # ga params
    num_generations = 100
    population_size = 100

    # aco params
    initial_temp = 1000
    cooling_rate = 0.85

    reuters_df = pd.read_csv('./dataset/Input/reuters_dataset.csv', encoding='utf-8')
    le = LabelEncoder()
    true_labels = le.fit_transform(reuters_df['label'])

    # spectral embedding
    Y = spectral_embedding(reuters_df, k)

    # experiments algorithms - metaheuristic algorithms(pso, ga, aco)
    pso_labels, pso_scores = pso(Y, k, num_particles, max_iter, inertia_weight, c1, c2)
    ga_labels, ga_scores = ga(Y, k, max_iter, population_size)
    sa_labels, sa_scores = SimulatedAnnealing(Y, k, initial_temp, cooling_rate, max_iter).run()

    algorithm_type = ['pso', 'ga', 'sa']
    cluster_labels = [pso_labels, ga_labels, sa_labels]
    silhouette_scores = [pso_scores, ga_scores, sa_scores]

    # 성능 평가
    for algorithm, cluster_label, score in zip(algorithm_type,  cluster_labels, silhouette_scores):
        accuracy, nmi, ari = evaluate_algorithms(true_labels, cluster_label)
        print(f"{algorithm} Accuracy: {accuracy}")
        print(f"{algorithm} NMI: {nmi}")
        print(f"{algorithm} silhouette_score: {score}")