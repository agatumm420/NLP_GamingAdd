import spacy
from sklearn import svm

from Proper_Analysis.reading_data import df


class Category:
    low_score = '7-10 points (low score)'
    medium_score = '11-15 points (medium score)'
    high_score = '16-26 points (large score'


train_y = []

for item in df['GAD_score']:
    if item <= 10:
        train_y.append(Category.low_score)
    elif item <= 15:
        train_y.append(Category.medium_score)
    else:
        train_y.append(Category.high_score)

train_x = list(df['NLP_1'])

nlp = spacy.load('pl_core_news_lg')

docs = [nlp(text) for text in train_x]
train_x_word_vectors = [x.vector for x in docs]

clf_svm_wv = svm.SVC(kernel='linear')
clf_svm_wv.fit(train_x_word_vectors, train_y)


def test_phrase(string):
    test_x = [string]
    test_docs = [nlp(text) for text in test_x]
    test_x_word_vectors = [x.vector for x in test_docs]
    result = clf_svm_wv.predict(test_x_word_vectors)
    print(result)

