from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from Proper_Analysis.reading_data import df
from sklearn.metrics import accuracy_score

N = 16
X_STRING = 'NLP_2'
Y_STRING = 'GDT_score'

x = df[X_STRING]
y = df[Y_STRING]


model_args = ClassificationArgs()
model_args.num_train_epochs = 2
model_args.regression = True

model = ClassificationModel(
    "herbert",
    "allegro/herbert-base-cased",
    num_labels=1,
    args=model_args,
    use_cuda=False
)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


train_df = pd.DataFrame({"text": x_train, "points": y_train})
eval_df = pd.DataFrame({"text": x_test, "points": y_test})

model.train_model(train_df)


result, model_outputs, wrong_predictions = model.eval_model(eval_df)

print(result)

