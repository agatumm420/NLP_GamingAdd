# kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1
from sklearn import svm
from Proper_Analysis.helpers import run_test_loops

FILE_NAME = 'TEST_TEST_1.xlsx'

N = 20
ACCEPTABLE_OFFSET = 2

X_STRINGS = ['NLP_well_being', 'NLP_gaming']
Y_STRINGS = ['GDT_score', 'GAD_score']
SVR_SETUPS = [
              svm.SVR(kernel='linear'),
              svm.SVR(kernel='linear', C=0.0001)
             ]

run_test_loops(FILE_NAME, N, X_STRINGS, Y_STRINGS, ACCEPTABLE_OFFSET, SVR_SETUPS)
