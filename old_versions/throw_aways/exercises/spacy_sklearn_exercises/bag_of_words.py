from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm


class Category:
    BOOKS = 'BOOKS'
    CLOTHING = 'CLOTHING'


train_x = ['I love the book', 'This is a great book', 'The fit is great', 'I love the shoes']
train_y = [Category.BOOKS, Category.BOOKS, Category.CLOTHING, Category.CLOTHING]

vectorizer = CountVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)

# Printing vectors
print(vectorizer.get_feature_names_out())
print('\n')
print(train_x_vectors.toarray())

clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_x_vectors, train_y)
test_x = vectorizer.transform(['Hero of the book had a great shoes'])

result = clf_svm.predict(test_x)
print(result)
