import re
import nltk
import warnings
warnings.filterwarnings('ignore')

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import eigsh

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
stopWords = stopwords.words('english')
charfilter = re.compile('[a-zA-Z]+')

def simple_tokenizer(text):
    #tokenizing the words:
    words = word_tokenize(text)
    #converting all the tokens to lower case:
    words = map(lambda word: word.lower(), words)
    #let's remove every stopwords
    words = [word for word in words if word not in stopWords]
    #stemming all the tokens
    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))
    ntokens = list(filter(lambda token : charfilter.match(token),tokens))
    return ntokens

# 유사도 그래프 정의
def define_similarity_graph(S):
    return cosine_similarity(S)

# 대각 행렬 정의
def define_diagonal_matrix(S):
    return np.diag(np.sum(S, axis=1))

# 라플라시안 행렬 구성
def construct_laplacian_matrix(D, W):
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    return D_inv_sqrt @ W @ D_inv_sqrt

# 가장 큰 고유 벡터 찾기
def find_largest_eigen_vectors(LM, k):
    _, eig_vectors = eigsh(LM, k=k, which='LM')
    return eig_vectors

# 행렬 Y 구성
def construct_matrix_Y(V):
    return V / np.linalg.norm(V, axis=1, keepdims=True)

def spectral_embedding(D, k):
    tf_idf_vector = TfidfVectorizer(ngram_range=(1, 1), tokenizer=simple_tokenizer)
    transformed_vector = tf_idf_vector.fit_transform(D['text'])
    print("DTM 크기:", transformed_vector.shape)

    W = define_similarity_graph(transformed_vector)
    D_matrix = define_diagonal_matrix(W)
    LM = construct_laplacian_matrix(D_matrix, W)
    V = find_largest_eigen_vectors(LM, k)
    Y = construct_matrix_Y(V)
    return Y