import spacy
from sklearn import svm
from Proper_Analysis.reading_data import df


class Category:
    low_score = '4-6 points'
    medium_score = '7-11 points'
    high_score = '12-19 points'


train_y = []

for item in df['GDT_score'][10:]:
    if item <= 6:
        train_y.append(Category.low_score)
    elif item <= 11:
        train_y.append(Category.medium_score)
    else:
        train_y.append(Category.high_score)

train_x = list(df['NLP_2'][10:])

nlp = spacy.load('pl_core_news_lg')

docs = [nlp(text) for text in train_x]
train_x_word_vectors = [x.vector for x in docs]

clf_svm_wv = svm.SVC(kernel='linear')
clf_svm_wv.fit(train_x_word_vectors, train_y)


def test_phrase(index):
    test_x = [df['NLP_2'][df.index[index]]]
    print(f'Real score is {df["GDT_score"][df.index[index]]}')
    test_docs = [nlp(text) for text in test_x]
    test_x_word_vectors = [x.vector for x in test_docs]
    result = clf_svm_wv.predict(test_x_word_vectors)
    print(result)


test_phrase(0)
test_phrase(1)
test_phrase(2)
test_phrase(3)
test_phrase(4)
test_phrase(5)
test_phrase(6)
test_phrase(7)
test_phrase(8)
test_phrase(9)





