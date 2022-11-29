# kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1
from sklearn import svm
from old_versions.Proper_Analysis.Transformers_Models.regression_herBERT import regression_with_herBERT
from old_versions.Proper_Analysis.Transformers_Models.regression_herBERT_2 import regression_with_herBERT_2
from old_versions.Proper_Analysis.helpers import run_test_loops

FILE_NAME = 'NLP-GAD_PHQ'

N = 100
ACCEPTABLE_OFFSET = 2

X_STRINGS = ['nlp_2', 'nlp_3', 'nlp_4', 'nlp_5', 'nlp_6']
Y_STRINGS = ['harmony_score']
SVR_SETUPS = [svm.SVR(kernel='linear')]
FUNCTIONS = [regression_with_herBERT]

run_test_loops(FILE_NAME, N, X_STRINGS, Y_STRINGS, ACCEPTABLE_OFFSET, SVR_SETUPS, FUNCTIONS)
