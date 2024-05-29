import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from func import simple_tokenizer, calculate_accuracy, normalized_mutual_info_score_v2, compute_ari
from scpso_algorithms import spectral_clustering_with_pso

k = 15
num_particles = 30
max_iter = 100
inertia = 0.5
c1 = 1
c2 = 1

df = pd.read_csv('dataset/Input/reuters_dataset.csv')
tf_idf_vector = TfidfVectorizer(tokenizer = simple_tokenizer, norm = 'l2', max_features=1000)
transformed_vector = tf_idf_vector.fit_transform(df['text'])
print("DTM 크기:", transformed_vector.shape)

cluster_labels = spectral_clustering_with_pso(transformed_vector, k, num_particles, max_iter, inertia, c1, c2)

le = LabelEncoder()
true_labels = le.fit_transform(df['label'])

results_df = pd.DataFrame({'True_Labels': true_labels, 'Cluster_Labels': cluster_labels})
results_df.to_csv('dataset/Output/labels_clusters.csv', index=False, encoding = 'utf-8')

# 성능 평가
accuracy = calculate_accuracy(true_labels, cluster_labels)
nmi = normalized_mutual_info_score_v2(true_labels, cluster_labels)
ari = compute_ari(true_labels, cluster_labels)

print(f"Accuracy: {accuracy}")
print(f"NMI: {nmi}")
print(f"ARI: {ari}")