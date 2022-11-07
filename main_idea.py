class Test:  
    def __init__(self, article, abstract):  
        self.article = article  
        self.abstract = abstract 



test_file = open('finished_files/test.txt', 'r') 
count = 0
test_lst = []
while True:   
    line = test_file.readline() 
    if not line: 
        break
    lst = line.split('abstract=')
    # remove <s> and </s>
    cleanAbstract = lst[1].replace("<s>", "").replace("</s>", "").strip()
    test_lst.append(Test(lst[0].split('article=')[1], cleanAbstract))
    count += 1
print(count)
test_file.close() 

print(test_lst[10].abstract)
print(test_lst[10].article)

import spacy
import pytextrank

nlp = spacy.load("en_core_web_sm")

# add PyTextRank to the spaCy pipeline
tr = pytextrank.TextRank()
nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

doc = nlp(test_lst[0].article)

# examine the top-ranked phrases in the document
for p in doc._.phrases:
    print("{:.4f} {:5d}  {}".format(p.rank, p.count, p.text))
    print(p.chunks)

import graphviz

keys = list(tr.seen_lemma.keys())
dot = graphviz.Digraph()

for node_id in tr.lemma_graph.nodes():
    text = keys[node_id][0].lower()
    rank = tr.ranks[node_id]
    label = "{} ({:.4f})".format(text, rank)
    print("{} {} ({:.4f})".format(node_id, text, rank))
    dot.node(str(node_id), label)

for edge in tr.lemma_graph.edges():
    dot.edge(str(edge[0]), str(edge[1]), constraint="false")
dot

import nltk
from nltk.tokenize import sent_tokenize
tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
reference_sentences = tokenizer.tokenize(test_lst[0].abstract)
print(reference_sentences)
print(len(reference_sentences))

pytextrankSum = ""
for sent in doc._.textrank.summary(limit_phrases=10, limit_sentences=len(reference_sentences)):
    pytextrankSum = pytextrankSum + " " + str(sent)

print(pytextrankSum)
print("\n----\n")
print(test_lst[0].abstract)

import spacy
import pytextrank

nlp = spacy.load("en_core_web_sm")

# add PyTextRank to the spaCy pipeline
tr = pytextrank.TextRank()
nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

def get_sum(tokenizer, article, abstract):
    doc = nlp(article)
    reference_sentences = tokenizer.tokenize(abstract)
    pytextrankSum = ""
    for sent in doc._.textrank.summary(limit_phrases=10, limit_sentences=len(reference_sentences)):
        pytextrankSum = pytextrankSum + " " + str(sent)
    return pytextrankSum  