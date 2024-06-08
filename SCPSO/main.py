import pandas as pd

from sklearn.preprocessing import LabelEncoder

from text_vectorization import spectral_embedding
from measure import calculate_accuracy, normalized_mutual_info_score_v2, compute_ari
from algorithms import pso, ga

# global parms
k = 15

# pso parms
num_particles = 30
max_iter = 100
inertia_weight = 0.5
c1 = 1
c2 = 1

# ga parms
num_generations = 100
population_size = 100

df = pd.read_csv('./dataset/Input/reuters_dataset_single_category_filtered.csv', encoding='utf-8')
le = LabelEncoder()
true_labels = le.fit_transform(df['label'])

# spectral embedding
Y = spectral_embedding(df, k)

# experiments algorithms - metaheuristic algorithms
pso_labels = pso(Y, k, num_particles, max_iter, inertia_weight, c1, c2)
ga_labels = ga(Y, k, num_generations, population_size)

pso_df = pd.DataFrame({'True_Labels': true_labels, 'Cluster_Labels': pso_labels})
ga_df = pd.DataFrame({'True_Labels': true_labels, 'Cluster_Labels': ga_labels})

# 성능 평가
accuracy = calculate_accuracy(true_labels, ga_labels)
nmi = normalized_mutual_info_score_v2(true_labels, ga_labels)
ari = compute_ari(true_labels, ga_labels)

print(f"Accuracy: {accuracy}")
print(f"NMI: {nmi}")
print(f"ARI: {ari}")