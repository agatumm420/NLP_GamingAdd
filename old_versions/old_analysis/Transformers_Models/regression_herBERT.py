import numpy as np
from transformers import AutoTokenizer, AutoModel
from old_versions.Proper_Analysis.Transformers_Models.helpers import regression_results
from old_versions.Proper_Analysis.reading_data.reading_final_data import model_names, df


def regression_with_herBERT(n, x_string, y_string, offset, SVR_setup):

    proper_df = df[~df[x_string].isnull()]

    tokenizer = AutoTokenizer.from_pretrained(model_names["herbert-base-cased"]["tokenizer"])
    nlp = AutoModel.from_pretrained(model_names["herbert-base-cased"]["model"])
    
    docs = [nlp(**tokenizer.batch_encode_plus(list(proper_df[x_string]), padding="longest",
                                              add_special_tokens=True,
                                              return_tensors="pt", ))]
    
    all_x = np.asarray([x.to_tuple()[0].detach().numpy() for x in docs])[0]
    num, x1, x2 = all_x.shape
    all_x_2d = all_x.reshape(num, x1 * x2)
    
    train_x = all_x_2d[n:]
    train_y = proper_df[y_string][n:]
    test_x = all_x_2d[:n]
    result_y = proper_df[y_string][:n]

    clf_svm_wv = SVR_setup
    clf_svm_wv.fit(train_x, train_y)
    
    result = clf_svm_wv.predict(test_x)

    print(f'{x_string}/{y_string}\nFUNC_1 {SVR_setup}')
    efficiency_dict = regression_results(result_y, result, offset)
    print(f'{x_string}/{y_string}\nFUNC_1 {SVR_setup}\n\n')

    return efficiency_dict
