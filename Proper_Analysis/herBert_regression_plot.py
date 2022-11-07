import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn import svm
from Proper_Analysis.reading_data import model_names, df

N = 0
X_STRING = 'NLP_1'
Y_STRING = 'GDT_score'
ACCEPTABLE_OFFSET = 2

tokenizer = AutoTokenizer.from_pretrained(model_names["herbert-base-cased"]["tokenizer"])
nlp = AutoModel.from_pretrained(model_names["herbert-base-cased"]["model"])

docs = [nlp(**tokenizer.batch_encode_plus(list(df[X_STRING]), padding="longest",
                                          add_special_tokens=True,
                                          return_tensors="pt", ))]

all_x = np.asarray([x.to_tuple()[0].detach().numpy() for x in docs])[0]
n, x1, x2 = all_x.shape
all_x_2d = all_x.reshape(n, x1 * x2)

predicted_scores = []

for iteration in range(8):
    N += 10

    train_x = list(all_x_2d[0:N-10]) + list(all_x_2d[N:])
    train_y = list(df[Y_STRING][0:N-10]) + list(df[Y_STRING][N:])
    test_x = all_x_2d[N-10:N]

    clf_svm_wv = svm.SVR(kernel="linear")
    clf_svm_wv.fit(train_x, train_y)

    result = clf_svm_wv.predict(test_x)

    predicted = 0
    failed = 0

    for score, prediction in zip((df[Y_STRING][:N]), result):
        score = int(score)

        predicted_scores.append(prediction)

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
    print(f'Success rate: {success_rate}\n\n')

print(predicted_scores)
