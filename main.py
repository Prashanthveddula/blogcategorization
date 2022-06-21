import nltk
from nltk.tree import Tree


nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")

# open files
article = open("docs/article3.txt", "r").read()
countries = open("docs/countries.txt", "r").read()

Indian = open("docs/India.txt", "r").read()
American = open("docs/USA.txt", 'r').read()
Japanese = open("docs/Japan.txt", 'r').read()
Russian = open("docs/Russia.txt", 'r').read()



# return a list of all country names and nationalities('Ameriacan', 'Indian')
def extract_countries(text):
    sent = nltk.tokenize.word_tokenize(text)
    pos_tag = nltk.pos_tag(sent)
    nes = nltk.ne_chunk(pos_tag)
    places = []

    for ne in nes:
        if type(ne) is nltk.tree.Tree:
            if ne.label() == "GPE":
                places.append(u" ".join([i[0] for i in ne.leaves()]))

    return places


# returns a list of all humans names in text
def extract_names(text):
    sent = nltk.word_tokenize(text)
    pos_tag = nltk.pos_tag(sent)
    nes = nltk.ne_chunk(pos_tag)
    res = []
    for ne in nes:
        if type(ne) == Tree:
            if ne.label() == "PERSON":
                res.append(u" ".join([i[0] for i in ne.leaves()]))

    return res

#pytest for extracting names
def test_answer1():
    assert extract_names(Indian) == ['Satya', 'Narayana Nadella']
def test_answer2():
    assert extract_names(American) == ['Marques', 'Keith Brownlee']
def test_answer3():
    assert extract_names(Japanese) == ['Makoto', 'Makoto Shinkai']
def test_answer4():
    assert extract_names(Russian) == ['Garry', 'Kimovich Kasparov']

if __name__ == "__main__":
    print(extract_names(Indian))
    print(extract_names(American))
    print(extract_names(Japanese))
    print(extract_names(Russian))
