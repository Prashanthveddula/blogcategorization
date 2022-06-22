import nltk


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
article2 = open("docs/test.txt", 'r').read()
dict = {}
with open("docs/name2lang.txt") as f:
    for line in f:
        temp = line.split(",")
        x = temp[0].strip()
        y = temp[1].strip()
        dict[x] = y



# Return a list of all country names and nationalities('Ameriacan', 'Indian')
def extract_countries(text):
    sent = nltk.tokenize.word_tokenize(text)
    pos_tag = nltk.pos_tag(sent)
    nes = nltk.ne_chunk(pos_tag)
    places = []

    for ne in nes:
        if type(ne) == nltk.tree.Tree:
            if ne.label() == "GPE":
                places.append(u" ".join([i[0] for i in ne.leaves()]))

    return places


# Returns a list of all humans names in text
def extract_names(text):
    sent = nltk.word_tokenize(text)
    pos_tag = nltk.pos_tag(sent)
    nes = nltk.ne_chunk(pos_tag)
    res = []
    for ne in nes:
        if type(ne) == nltk.tree.Tree:
            if ne.label() == "PERSON":
                res.append(u" ".join([i[0] for i in ne.leaves()]))

    return res


#Extracts humans names and returns their respective country
def name_to_country(text):
    names = []
    nametoc = []
    nes = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)))
    for n in nes:
        if type(n) == nltk.tree.Tree:
            if n.label() == "PERSON":
                name = f" ".join(i[0] for i in n.leaves())
                names.append(name)
    for j in names:
        temp = j.split(" ")
        found = False
        for k in temp:
            if k in dict:
                nametoc.append((j, dict[k]))
                found = True
                break
        if found == False:
            nametoc.append((j, 'Not identified'))
    return nametoc


#pytest for extracting names
def test_answer1():
    assert extract_names(Indian) == ['Satya', 'Narayana Nadella']
def test_answer2():
    assert extract_names(American) == ['Marques', 'Keith Brownlee']
def test_answer3():
    assert extract_names(Japanese) == ['Makoto', 'Makoto Shinkai']
def test_answer4():
    assert extract_names(Russian) == ['Garry', 'Kimovich Kasparov']


#driver code
if __name__ == "__main__":
    print(extract_names(Indian))
    print(extract_names(American))
    print(extract_names(Japanese))
    print(extract_names(Russian))
    print(name_to_country(article2))
