# scrapping data
import requests
from bs4 import BeautifulSoup


class Category:
    movie_quote = 'Movie quote'
    advert_slogan = 'Advert slogan'


URL_1 = "https://en.wikiquote.org/wiki/Advertising_slogans"
URL_2 = "https://en.wikipedia.org/wiki/AFI%27s_100_Years...100_Movie_Quotes"

r = requests.get(URL_2)

soup = BeautifulSoup(r.content, 'html5lib')


train_x = []
train_y = []

# Populating movies
for x in soup.findAll('tr')[15:115]:
    for y in x:
        proper_y = str(y)

        if 'title' in proper_y:

            if proper_y[6] == 'a':
                updated_string = proper_y.split('title="')[1]
                final_string = updated_string.split('">')[0]
                train_x.append(final_string)
                train_y.append(Category.movie_quote)
                break
        else:
            if '<td>"' in proper_y:
                train_x.append(proper_y.split('<td>"')[1].split('"')[0])
                train_y.append(Category.movie_quote)
                break

train_x[89] = train_x[89].split('i>. ')[1]

# Populating slogans
r = requests.get(URL_1)

soup = BeautifulSoup(r.content, 'html5lib')

for x in soup.findAll('tr')[1:250]:
    train_x.append(x.findAll('td')[0].text)
    train_y.append(Category.advert_slogan)
