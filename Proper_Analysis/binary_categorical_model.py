import spacy
from sklearn import svm
from Proper_Analysis.reading_data import df

N = 16
X_STRING = 'NLP_6'
Y_STRING = 'GAD_score'
THRESHOLD = 14


class Category:
    non_problematic = 'Non problematic'
    problematic = 'Problematic'


def categorize(arr):
    categorized_list = []
    for item in arr:
        if item < THRESHOLD:
            categorized_list.append(Category.non_problematic)
        elif item >= THRESHOLD:
            categorized_list.append(Category.problematic)
        else:
            print(f'Error with {item}')
    return categorized_list


train_x = df[X_STRING][N:]
train_y = categorize(df[Y_STRING][N:])

nlp = spacy.load('pl_core_news_lg')

docs = [nlp(text) for text in train_x]
train_x_word_vectors = [x.vector for x in docs]

clf_svm_wv = svm.SVC(kernel='linear')
clf_svm_wv.fit(train_x_word_vectors, train_y)

test_x = df[X_STRING][:N]
real_y = categorize(df[Y_STRING][:N])
real_score = df[Y_STRING][:N]

test_docs = [nlp(text) for text in test_x]
test_x_word_vectors = [x.vector for x in test_docs]
result = clf_svm_wv.predict(test_x_word_vectors)

predicted = 0
failed = 0

for prediction, real_classification, real_score in zip(result, real_y, real_score):

    if prediction == real_classification:
        predicted += 1
        if real_classification == 'Non problematic':
            continue

        print('PREDICTED')

    else:
        print('FAILED')
        print(f'by {abs(THRESHOLD - real_score)} points')
        failed += 1

    print(f'Prediction: {prediction}, True state: {real_classification} | Points: {real_score}\n')


success_rate = predicted / (predicted + failed) * 100
print(f'Predicted: {predicted}')
print(f'Failed: {failed}')
print(f'Success rate: {success_rate}')
