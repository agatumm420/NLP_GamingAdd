from sklearn.pipeline import Pipeline
from transformers import AutoTokenizer, AutoModel
from Proper_Analysis.Transformers_Models.helpers import BertTransformer, regression_results
from Proper_Analysis.reading_data import model_names, df


def regression_with_herBERT_2(n, x_string, y_string, offset, SVR_setup):
    tokenizer = AutoTokenizer.from_pretrained(model_names["herbert-base-cased"]["tokenizer"])
    model = AutoModel.from_pretrained(model_names["herbert-base-cased"]["model"])
    bert_transformer = BertTransformer(tokenizer, model)
    classifier = SVR_setup
    model = Pipeline(
        [
            ("vectorizer", bert_transformer),
            ("classifier", classifier),
        ]
    )

    train_x = df[x_string][n:]
    train_y = df[y_string][n:]
    test_x = df[x_string][:n]
    result_y = df[y_string][:n]

    model.fit(train_x, train_y)
    result = model.predict(test_x)

    print(f'{x_string}/{y_string}\nFUNC_2 {SVR_setup}')
    efficiency_dict = regression_results(result_y, result, offset)
    print(f'{x_string}/{y_string}\nFUNC_2 {SVR_setup}\n\n')

    return efficiency_dict
