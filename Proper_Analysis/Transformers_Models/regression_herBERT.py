import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn import svm
from Proper_Analysis.helpers import regression_results
from Proper_Analysis.reading_data import model_names, df


def regression_with_herBERT(n, x_string, y_string, offset, kernel):
    tokenizer = AutoTokenizer.from_pretrained(model_names["herbert-base-cased"]["tokenizer"])
    nlp = AutoModel.from_pretrained(model_names["herbert-base-cased"]["model"])
    
    docs = [nlp(**tokenizer.batch_encode_plus(list(df[x_string]), padding="longest",
                                              add_special_tokens=True,
                                              return_tensors="pt", ))]
    
    all_x = np.asarray([x.to_tuple()[0].detach().numpy() for x in docs])[0]
    num, x1, x2 = all_x.shape
    all_x_2d = all_x.reshape(num, x1 * x2)
    
    train_x = all_x_2d[n:]
    train_y = df[y_string][n:]
    test_x = all_x_2d[:n]
    
    clf_svm_wv = svm.SVR(kernel=kernel)
    clf_svm_wv.fit(train_x, train_y)
    
    result = clf_svm_wv.predict(test_x)
    
    efficiency_dict = regression_results(df[y_string][:n], result, offset)
    return efficiency_dict
