import spacy
from scraping_data import train_x, train_y
from sklearn import svm

nlp = spacy.load('en_core_web_md')

docs = [nlp(text) for text in train_x]
train_x_word_vectors = [x.vector for x in docs]

clf_svm_wv = svm.SVC(kernel='linear')
clf_svm_wv.fit(train_x_word_vectors, train_y)


test_x = ['No, I am your father.', 'Just Do It.', ' Because You\'re Worth It.',
          'Betcha Can’t Eat Just One', 'You\'re gonna need a bigger boat',
          'The happiest place on Earth.', 'The original. If your grandfather hadn’t worn it, you wouldn’t exist.',
          'A diamond is forever.', 'MasterCard: “There are some things money can’t buy. For everything else, there’s MasterCard.',
          'Release the Kraken!', 'I\'m already pregnant. So, what other shenanigans could I get myself into?',
          'Which would be worse, to live a monster or die as a good man?',
          'I know who I am. I\'m the dude playing a dude disguised as another dude!', '"That\'s a bingo!"',
          "I'm going to have to science the s*** out of this.", '"That\'s my secret, Captain: I\'m always angry."',
          'I live my life a quarter mile at a time.', 'Nobody makes me bleed my own blood. Nobody!',
          'Very nice!', 'I could do this all day', 'Do what you can’t.',
          'Our blades are f***ing great.” and “Shave time. Shave money.', 'The ultimate driving machine',
          'This is the day you will always remember as the day you almost caught Captain Jack Sparrow!',
          'Not my tempo.', 'What’s in your wallet?', 'Reduce your carbon footprint in style.',
          'That was easy.', 'Rewards reimagined.', 'Snap! Crackle! Pop!', 'We\'re goin\' streaking!',
          "I'm the guy who does his job. You must be the other guy.",
          "What is this? A center for ants? How can we be expected to teach children to learn how to read...if"
          " they can't even fit inside the building?",
          "Look at me. I'm the captain now.", "I'm just one stomach flu away from my goal weight.", 'Is it in you?',
          'Let us guide you home.', 'Belong anywhere.', 'I am Groot.', 'I drink your milkshake. I drink it up.',
          'I am a golden god!', 'My precious.', 'Can you hear me now? Good', 'America runs on dunkin.', 'Save money. Live better',
          'Get off my lawn.', 'I have nipples Greg. Could you milk me?', 'Let’s go places.'
        ]

test_x_answers = ['Movie quote', 'Advert slogan', 'Advert slogan', 'Advert slogan', 'Movie quote',
                  'Advert slogan', 'Advert slogan', 'Advert slogan', 'Advert slogan', 'Movie quote', 'Movie quote',
                  'Movie quote', 'Movie quote', 'Movie quote', 'Movie quote', 'Movie quote', 'Movie quote', 'Movie quote',
                  'Movie quote', 'Movie quote', 'Advert slogan', 'Advert slogan', 'Advert slogan', 'Movie quote', 'Movie quote',
                  'Advert slogan', 'Advert slogan', 'Advert slogan', 'Advert slogan', 'Advert slogan', 'Movie quote',
                  'Movie quote', 'Movie quote', 'Movie quote', 'Movie quote', 'Advert slogan', 'Advert slogan', 'Advert slogan',
                  'Movie quote', 'Movie quote', 'Movie quote', 'Movie quote', 'Advert slogan', 'Advert slogan', 'Advert slogan',
                  'Movie quote', 'Movie quote', 'Advert slogan'
                  ]

test_docs = [nlp(text) for text in test_x]
test_x_word_vectors = [x.vector for x in test_docs]

result = clf_svm_wv.predict(test_x_word_vectors)

predicted = 0
failed = 0
for x, y, z in zip(test_x, result, test_x_answers):
    if y == z:
        print('Successfully predicted')
        predicted += 1
    else:
        print('Failure')
        failed += 1
    print(f'Guess: {y} | Answer: {z} | Text: {x}')
    print('\n')

success_rate = predicted / (predicted + failed) * 100
print(f'Predicted: {predicted}')
print(f'Failed: {failed}')
print(f'Success rate: {success_rate} %')
