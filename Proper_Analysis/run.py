# kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1
from sklearn import svm
from Proper_Analysis.Transformers_Models.regression_herBERT import regression_with_herBERT
from Proper_Analysis.Transformers_Models.regression_herBERT_2 import regression_with_herBERT_2
from Proper_Analysis.helpers import run_test_loops

FILE_NAME = 'TESTING_DIFFERENT_FUNCTIONS'

N = 20
ACCEPTABLE_OFFSET = 2

X_STRINGS = ['NLP_gaming', 'NLP_all', 'NLP_2', 'NLP_4']
Y_STRINGS = ['GDT_score', 'GAD_score', 'PHQ_score']
SVR_SETUPS = [svm.SVR(kernel='linear')]
FUNCTIONS = [regression_with_herBERT, regression_with_herBERT_2]

run_test_loops(FILE_NAME, N, X_STRINGS, Y_STRINGS, ACCEPTABLE_OFFSET, SVR_SETUPS, FUNCTIONS)
