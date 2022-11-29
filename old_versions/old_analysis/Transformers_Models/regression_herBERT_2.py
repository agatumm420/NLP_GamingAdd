from sklearn.pipeline import Pipeline
from transformers import AutoTokenizer, AutoModel
from old_versions.Proper_Analysis.Transformers_Models.helpers import BertTransformer, regression_results
from old_versions.Proper_Analysis.reading_data.reading_final_data import model_names, df


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

    proper_df = df[~df[x_string].isnull()]

    train_x = proper_df[x_string][n:]
    train_y = proper_df[y_string][n:]
    test_x = proper_df[x_string][:n]
    result_y = proper_df[y_string][:n]

    model.fit(train_x, train_y)
    result = model.predict(test_x)

    print(f'{x_string}/{y_string}\nFUNC_2 {SVR_setup}')
    efficiency_dict = regression_results(result_y, result, offset)
    print(f'{x_string}/{y_string}\nFUNC_2 {SVR_setup}\n\n')

    return efficiency_dict
