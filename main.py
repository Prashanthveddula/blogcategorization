import nltk
import os
from nltk import tokenize

#open files
f1 = open("article2.txt")
f2 = open("countries.txt")
article = f1.read()
countries = f2.read()

#returns a list of all country names from input text.
def extract_countries(text, CountriesList):
    res = []
    sent = nltk.word_tokenize(text)
    pos_tag = nltk.pos_tag(sent)
    
    for (word, pos) in pos_tag:
        if (pos[:2] == 'NN') and (word in CountriesList):
            res.append(word) 
    return res


if __name__ == '__main__':
    print(extract_countries(article, countries))
