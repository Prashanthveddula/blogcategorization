import nltk
from nltk import tokenize
from nltk.tree import Tree

nltk.download('words')

#open files
article = open("docs/article3.txt", 'r').read()
countries = open("docs/countries.txt", 'r').read()

Indian = open("docs/India.txt", 'r').read()
American = open("docs/USA.txt", 'r').read()
Chinese = open("docs/China.txt", 'r').read()
Russian = open("docs/Russia.txt", 'r').read()


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
    sent = nltk.tokenize.word_tokenize(text)
    pos_tag = nltk.pos_tag(sent)
    nes = nltk.ne_chunk(pos_tag)
    places = []
    for ne in nes:
        if type(ne) is nltk.tree.Tree:
            if (ne.label() == 'GPE'):
                places.append(u' '.join([i[0] for i in ne.leaves()]))
        
    return places

#returns a list of all humans names in text
def extract_names(text):
    sent = nltk.word_tokenize(text)
    pos_tag = nltk.pos_tag(sent)
    nes = nltk.ne_chunk(pos_tag)
    res = []
    for ne in nes:
        if type(ne) == Tree:
            if (ne.label() == 'PERSON'):
                res.append(u' '.join([i[0] for i in ne.leaves()]))

    return res


if __name__ == '__main__':
    print(extract_countries(article, countries))
    print(extract_countries2(article))
    print(extract_names(Indian))
    print(extract_names(American))
    print(extract_names(Chinese))
    print(extract_names(Russian))
