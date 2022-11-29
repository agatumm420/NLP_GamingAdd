import spacy
from old_versions.Proper_Analysis.helpers import regression_results
from old_versions.Proper_Analysis.reading_data.reading_pilot_data import df


def regression_with_spaCy(n, x_string, y_string, offset, SVR_setup):
    train_x = df[x_string][n:]
    train_y = df[y_string][n:]

    nlp = spacy.load('pl_core_news_lg')

    docs = [nlp(text) for text in train_x]
    train_x_word_vectors = [x.vector for x in docs]

    clf_svm_wv = SVR_setup
    clf_svm_wv.fit(train_x_word_vectors, train_y)

    test_x = df[x_string][:n]
    test_docs = [nlp(text) for text in test_x]
    test_x_word_vectors = [x.vector for x in test_docs]
    result = clf_svm_wv.predict(test_x_word_vectors)

    efficiency = regression_results(df[y_string][:n], result, offset)
    return efficiency
