import spacy
from sklearn import svm
from Proper_Analysis.helpers import categorize, print_categorical_results
from Proper_Analysis.reading_data import df

N = 16
X_STRING = 'NLP_6'
Y_STRING = 'GAD_score'
THRESHOLD = 14


train_x = df[X_STRING][N:]
train_y = categorize(df[Y_STRING][N:], THRESHOLD)

nlp = spacy.load('pl_core_news_lg')

docs = [nlp(text) for text in train_x]
train_x_word_vectors = [x.vector for x in docs]

clf_svm_wv = svm.SVC(kernel='linear')
clf_svm_wv.fit(train_x_word_vectors, train_y)

test_x = df[X_STRING][:N]
real_y = categorize(df[Y_STRING][:N], THRESHOLD)
real_score = df[Y_STRING][:N]

test_docs = [nlp(text) for text in test_x]
test_x_word_vectors = [x.vector for x in test_docs]
result = clf_svm_wv.predict(test_x_word_vectors)

print_categorical_results(real_y, result, real_score, THRESHOLD)
