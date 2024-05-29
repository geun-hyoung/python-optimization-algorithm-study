import nltk
from nltk.corpus import stopwords, reuters
from sklearn.datasets import fetch_20newsgroups
import pandas as pd

# dataset 1
nltk.download('reuters')
files = reuters.fileids()

# 데이터셋 생성
test_data = []
test_labels = []
train_data = []
train_labels = []
for file in reuters.fileids():
    if file.startswith('training/'):
        train_data.append(reuters.raw(file))
        # 첫 번째 라벨만 선택
        train_labels.append(reuters.categories(file)[0])
    elif file.startswith('test/'):
        test_data.append(reuters.raw(file))
        # 첫 번째 라벨만 선택
        test_labels.append(reuters.categories(file)[0])
    else:
        print('error')

# 모든 90개의 카테고리를 고려
our_labels = reuters.categories()

# Trainset 생성
docs_train = []
docs_labels = []
for i in range(len(train_labels)):
    if train_labels[i] in our_labels:
        docs_train.append(train_data[i])
        docs_labels.append(train_labels[i])

df_reuters = pd.DataFrame({'text': docs_train, 'label': docs_labels})
df_reuters.to_csv('./dataset/Inuput/reuters_dataset.csv', index=False, encoding='utf-8')

# # dataset 2
# newsgroups = fetch_20newsgroups(subset='all')
# texts = newsgroups.data
# labels = newsgroups.target
# label_names = [newsgroups.target_names[label] for label in labels]
#
# df_newsgroups = pd.DataFrame({'text': texts, 'label': label_names})
# df_newsgroups.to_csv('./dataset/20newsgroups_dataset.csv', index=False, encoding='utf-8')