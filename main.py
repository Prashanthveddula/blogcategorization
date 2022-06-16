import nltk
import os
from nltk import tokenize


#open files
f1 = open("docs/article1.txt")
f2 = open("docs/countries.txt")
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


#return a list of all country names and nationalities('Ameriacan', 'Indian')
def extract_countries2(text): 
    sent = nltk.tokenize.wordpunct_tokenize(text)
    pos_tag = nltk.pos_tag(sent)
    nes = nltk.ne_chunk(pos_tag)
    places = []
    for ne in nes:
        if type(ne) is nltk.tree.Tree:
            if (ne.label() == 'GPE'):
                places.append(u' '.join([i[0] for i in ne.leaves()]))
        
    return places


if __name__ == '__main__':
    print(extract_countries2(article))
