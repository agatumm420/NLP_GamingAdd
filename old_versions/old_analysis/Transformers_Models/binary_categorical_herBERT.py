import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn import svm
from old_versions.Proper_Analysis.helpers import print_categorical_results, categorize
from old_versions.Proper_Analysis.reading_data.reading_final_data import df, model_names

N = 100
X_STRING = 'nlp_2'
Y_STRING = 'GDT_score'
THRESHOLD = 12


class Category:
    non_problematic = 'Non problematic'
    problematic = 'Problematic'


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
train_y = categorize(df[Y_STRING][N:], THRESHOLD)
test_x = all_x_2d[:N]
real_y = categorize(df[Y_STRING][:N], THRESHOLD)
real_score = df[Y_STRING][:N]

clf_svm_wv = svm.SVC(kernel='linear')
clf_svm_wv.fit(train_x, train_y)

result = clf_svm_wv.predict(test_x)

print_categorical_results(real_y, result, real_score, THRESHOLD)
