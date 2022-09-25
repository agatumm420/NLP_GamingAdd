import tensorflow_hub as hub
import tensorflow_text as text

# Sources:
# https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4
# https://www.youtube.com/watch?v=7kLi8u2dJz0&ab_channel=codebasics

preprocess_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
encored_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'

bert_preprocess_model = hub.KerasLayer(preprocess_url)

text_test = ['Nice movie indeed', 'I love python programming']

text_preprocessed = bert_preprocess_model(text_test)
print(text_preprocessed.keys())
print(text_preprocessed)


bert_model = hub.KerasLayer(encored_url)

bert_results = bert_model(text_preprocessed)
print(bert_results.keys())

# Embedding for entire sequences
print(bert_results['pooled_output'])

print(bert_results)
