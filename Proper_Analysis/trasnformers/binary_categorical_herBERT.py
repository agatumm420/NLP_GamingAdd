import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn import svm
from Proper_Analysis.reading_data import df, model_names

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


tokenizer = AutoTokenizer.from_pretrained(model_names["herbert-base-cased"]["tokenizer"])
model = AutoModel.from_pretrained(model_names["herbert-base-cased"]["model"])

nlp = AutoModel.from_pretrained(model_names["herbert-base-cased"]["model"])
docs = [nlp(**tokenizer.batch_encode_plus(list(df[X_STRING]), padding="longest",
                                          add_special_tokens=True,
                                          return_tensors="pt", ))]

all_x = np.asarray([x.to_tuple()[0].detach().numpy() for x in docs])[0]
n, x1, x2 = all_x.shape
all_x_2d = all_x.reshape(n, x1 * x2)

train_x = all_x_2d[N:]
train_y = categorize(df[Y_STRING][N:])
test_x = all_x_2d[:N]
real_y = categorize(df[Y_STRING][:N])
real_score = df[Y_STRING][:N]

clf_svm_wv = svm.SVC(kernel='linear')
clf_svm_wv.fit(train_x, train_y)

result = clf_svm_wv.predict(test_x)

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
