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

# pip install stanza
import stanza
# stanza.download('en') 
nlp = stanza.Pipeline('en')
print(doc)
# print(doc.entities)

# https://github.com/stanfordnlp/stanza/blob/main/demo/Stanza_Beginners_Guide.ipynb
for i, sent in enumerate(en_doc.sentences):
    print("[Sentence {}]".format(i+1))
    for word in sent.words:
        print("{:12s}\t{:12s}\t{:6s}\t{:d}\t{:12s}".format(\
              word.text, word.lemma, word.pos, word.head, word.deprel))
    print("")
    
print("Mention text\tType\tStart-End")
for ent in en_doc.ents:
    print("{}\t{}\t{}-{}".format(ent.text, ent.type, ent.start_char, ent.end_char))
    
# LIGHTWEIGHT VERSION gets candidate entities directly from the Wiki API 
def get_wiki_entities(entry):
  wiki_url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={entry}&language=en&format=json"
  result_lst = requests.get(url).json()
  ent_lst = list(set([result['display']['label']['value'] for result in result_lst['search']]))
  return ent_lst

#please refer to https://github.com/einan/simit/blob/main/main_idea.py for the BFS-based DEP embedding model
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

#TODO: the lightweight version gets vector representations by using the REST API of the pre-trained language model

