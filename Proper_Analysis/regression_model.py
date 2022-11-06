import spacy
from sklearn import svm
from reading_data import df

N = 20
X_STRING = 'NLP_6'
Y_STRING = 'GDT_score'
ACCEPTABLE_OFFSET = 2


train_x = df[X_STRING][N:]
train_y = df[Y_STRING][N:]


nlp = spacy.load('pl_core_news_lg')

docs = [nlp(text) for text in train_x]
train_x_word_vectors = [x.vector for x in docs]

clf_svm_wv = svm.SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)
clf_svm_wv.fit(train_x_word_vectors, train_y)

test_x = df[X_STRING][:N]
test_docs = [nlp(text) for text in test_x]
test_x_word_vectors = [x.vector for x in test_docs]
result = clf_svm_wv.predict(test_x_word_vectors)

predicted = 0
failed = 0

for score, prediction in zip((df[Y_STRING][:N]), result):
    score = int(score)
    if round(prediction) in range(score - ACCEPTABLE_OFFSET, score + ACCEPTABLE_OFFSET):
        print(f'PREDICTED')
        predicted += 1
    else:
        print('FAILED')
        failed += 1
    print(f'Prediction: {round(prediction, 2)}, Real score: {score}')
    print(f'Off by {score - prediction}')
    print('\n')


success_rate = predicted / (predicted + failed) * 100
print(f'Predicted: {predicted}')
print(f'Failed: {failed}')
print(f'Success rate: {success_rate}')
