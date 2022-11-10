import spacy
from scraping_data import train_x, train_y
from sklearn import svm

nlp = spacy.load('en_core_web_md')

docs = [nlp(text) for text in train_x]
train_x_word_vectors = [x.vector for x in docs]

clf_svm_wv = svm.SVC(kernel='linear')
clf_svm_wv.fit(train_x_word_vectors, train_y)

test_x = ['Everybody makes mistakes. Everybody has those days.', "How many of these are just horrible mistakes I made? I mean, maybe I'd stop making so many if I let myself learn from them.",
          "Good instincts are earned by making mistakes.", "“The future is scary, but you can’t just run back to the past because it’s familiar. Yes it’s tempting, but it’s a mistake.”"
          ]

test_x_answers = [1, 1, 1, 1]

for i in range(len(test_x_answers)):
    if test_x_answers[i] == 1:
        test_x_answers[i] = 'Movie quote'
    elif test_x_answers[i] == 0:
        test_x_answers[i] = 'Advert slogan'

print(test_x_answers)

test_docs = [nlp(text) for text in test_x]
test_x_word_vectors = [x.vector for x in test_docs]

result = clf_svm_wv.predict(test_x_word_vectors)

predicted = 0
failed = 0

for text, guess, answer in zip(test_x, result, test_x_answers):
    if guess == answer:
        print('Successfully predicted')
        predicted += 1
    else:
        print('Failure')
        failed += 1
    print(f'Guess: {guess} | Answer: {answer} | Text: {text}')
    print('\n')

success_rate = predicted / (predicted + failed) * 100
print(f'Predicted: {predicted}')
print(f'Failed: {failed}')
print(f'Success rate: {success_rate} %')
