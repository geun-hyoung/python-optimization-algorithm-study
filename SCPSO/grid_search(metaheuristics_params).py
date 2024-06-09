import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid

from text_vectorization import spectral_embedding
from algorithms import pso, ga, SimulatedAnnealing

def run_grid_search(algorithm, params_grid, data, k):
    best_score = -1
    best_params = None
    best_labels = None
    max_iter = 100  # algorithms_iterations

    for params in ParameterGrid(params_grid):
        if algorithm == 'pso':
            labels, score = pso(data, k, max_iter, **params)
        elif algorithm == 'ga':
            labels, score = ga(data, k, **params)
        elif algorithm == 'sa':
            sa = SimulatedAnnealing(data, k, max_iter, **params)
            labels, score = sa.run()
        else:
            raise ValueError("Unsupported algorithm")

        if score > best_score:
            best_score = score
            best_params = params
            best_labels = labels

    return best_params, best_labels, best_score

if __name__ == "__main__":
    k = 15  # number_of_clusters
    max_iter = 100  # algorithms_iterations

    reuters_df = pd.read_csv('./dataset/Input/reuters_dataset.csv', encoding='utf-8')
    le = LabelEncoder()
    true_labels = le.fit_transform(reuters_df['label'])

    # spectral embedding
    Y = spectral_embedding(reuters_df, k)

    # 파라미터 그리드 정의
    pso_params_grid = {
        'num_particles': [20, 30, 40],
        'inertia_weight': [0.5, 0.7, 0.9],
        'c1': [1, 1.5, 2],
        'c2': [1, 1.5, 2]
    }

    ga_params_grid = {
        'population_size': [50, 100, 150],
        'num_generations': [50, 100, 150],
        'crossover_rate': [0.5, 0.6, 0.7],
        'mutation_rate': [0.1, 0.15, 0.2]
    }

    sa_params_grid = {
        'initial_temp': [100, 500, 1000],
        'cooling_rate': [0.85, 0.9, 0.95]
    }

    # 그리드 탐색 실행
    best_pso_params, best_pso_labels, best_pso_score = run_grid_search('pso', pso_params_grid, Y, k)
    best_ga_params, best_ga_labels, best_ga_score = run_grid_search('ga', ga_params_grid, Y, k)
    best_sa_params, best_sa_labels, best_sa_score = run_grid_search('sa', sa_params_grid, Y, k)

    # 최적 파라미터 출력
    print("PSO 최적 파라미터:", best_pso_params, "점수:", best_pso_score)
    print("GA 최적 파라미터:", best_ga_params, "점수:", best_ga_score)
    print("SA 최적 파라미터:", best_sa_params, "점수:", best_sa_score)