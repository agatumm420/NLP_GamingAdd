# pip install spacy
# python -m spacy download en_core_web_md

import spacy
from bag_of_words import train_x, train_y
from sklearn import svm

nlp = spacy.load('en_core_web_md')

docs = [nlp(text) for text in train_x]
train_x_word_vectors = [x.vector for x in docs]

clf_svm_wv = svm.SVC(kernel='linear')
clf_svm_wv.fit(train_x_word_vectors, train_y)


test_x = ['Your tie looks absolutely fabulous']
test_docs = [nlp(text) for text in test_x]
test_x_word_vectors = [x.vector for x in test_docs]

result = clf_svm_wv.predict(test_x_word_vectors)

print(result)
