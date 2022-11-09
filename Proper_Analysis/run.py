# kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1
from Proper_Analysis.Spacy_Models.regression_model import regression_with_spaCy
from Proper_Analysis.Transformers_Models.regression_herBERT import regression_with_herBERT

N = 20
X_STRING = 'NLP_6'
Y_STRING = 'PHQ_score'
ACCEPTABLE_OFFSET = 2
KERNEL = 'linear'

bert_efficiency = regression_with_herBERT(N, X_STRING, Y_STRING, ACCEPTABLE_OFFSET, KERNEL)

bert_efficiency_2 = regression_with_herBERT(N, 'NLP_1', Y_STRING, ACCEPTABLE_OFFSET, KERNEL)

print(f'Bert nlp6/GAD: {bert_efficiency}, Bert nlp1/GAD: {bert_efficiency_2}')

