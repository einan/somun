#Text Summarization dataset preparation
#https://machinelearningmastery.com/prepare-news-articles-text-summarization/

from os import listdir
import string

def load_doc(filename):
	file = open(filename, encoding='utf-8')
	text = file.read()
	file.close()
	return text

def split_story(doc):
	index = doc.find('@highlight')
	story, highlights = doc[:index], doc[index:].split('@highlight')
	highlights = [h.strip() for h in highlights if len(h) > 0]
	return story, highlights

def load_stories(directory):
	stories = list()
	for name in listdir(directory):
		filename = directory + '/' + name
		doc = load_doc(filename)
		story, highlights = split_story(doc)
		stories.append({'story':story, 'highlights':highlights})
	return stories

def clean_lines(lines):
	cleaned = list()
	table = str.maketrans('', '', string.punctuation)
	for line in lines:
		index = line.find('(CNN) -- ')
		if index > -1:
			line = line[index+len('(CNN)'):]
		line = line.split()
		line = [word.lower() for word in line]
		line = [w.translate(table) for w in line]
		line = [word for word in line if word.isalpha()]
		cleaned.append(' '.join(line))
	cleaned = [c for c in cleaned if len(c) > 0]
	return cleaned

stories = load_stories('cnn/stories/')
print('Loaded Stories %d' % len(stories))

for example in stories:
	example['story'] = clean_lines(example['story'].split('\n'))
	example['highlights'] = clean_lines(example['highlights'])
  
#text summarization methods
#https://towardsdatascience.com/text-summarization-in-python-76c0a41f0dc4

from gensim.summarization.summarizer import summarize

example = stories[0]
fullStr = '. '.join(example['story'])
print(summarize(fullStr)) #TextRank

from rouge import rouge_n_summary_level
from rouge import rouge_l_summary_level
from rouge import rouge_w_summary_level

def eval(tokenizer, ref_text, sum_text):
    reference_sentences = tokenizer.tokenize(ref_text)
    summary_sentences = tokenizer.tokenize(sum_text)
    _, _, rouge_1 = rouge_n_summary_level(summary_sentences, reference_sentences, 1)
    _, _, rouge_2 = rouge_n_summary_level(summary_sentences, reference_sentences, 2)
    _, _, rouge_l = rouge_l_summary_level(summary_sentences, reference_sentences)
    #_, _, rouge_w = rouge_w_summary_level(summary_sentences, reference_sentences)
    #print('ROUGE-W: %f' % rouge_w)
    return rouge_1, rouge_2, rouge_l

import re

REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}

def clean(x):
    return re.sub(
            r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
            lambda m: REMAP.get(m.group()), x)

def read_file(file_path):
    test_file = open(file_path, 'r') 
    count = 0
    test_lst = []
    while True:   
        line = test_file.readline() 
        if not line: 
            break
        lst = line.split('abstract=')
        # remove <s> and </s>
        cleanAbstract = clean(lst[1].replace("<s>", "").replace("</s>", "").strip())
        test_lst.append(Test(clean(lst[0].split('article=')[1]), cleanAbstract))
        count += 1
    print(count)
    test_file.close() 
    return test_lst

import nltk
from nltk.tokenize import sent_tokenize

def average(lst): 
    return sum(lst) / len(lst) 

def eval_All(file_path):
    lang = 'EN'
    method = 'TextRank'
    if(lang == 'EN'):
        tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
    if(lang == 'TR'):
        tokenizer = nltk.data.load('nltk:tokenizers/punkt/turkish.pickle')  
    test_lst = read_file('finished_files/chunked_test_txt/' + file_path)
    rouge_1_lst = []
    rouge_2_lst = []
    rouge_l_lst = []
    for t in test_lst:
        pytextrankSum = get_sum(tokenizer, t.article, t.abstract)
        rouge_1, rouge_2, rouge_l = eval(tokenizer, t.abstract, pytextrankSum)
        rouge_1_lst.append(rouge_1)
        rouge_2_lst.append(rouge_2)
        rouge_l_lst.append(rouge_l)
    print(method + " & " + lang + " & " + str(round(average(rouge_1_lst) * 100,2)) +" & " + str(round(average(rouge_2_lst) * 100,2)) + " & " + str(round(average(rouge_l_lst) * 100,2)) + "  \\\\")    
    results = {}
    results['rouge_1'] = round(average(rouge_1_lst) * 100,2)
    results['rouge_2'] = round(average(rouge_2_lst) * 100,2)
    results['rouge_l'] = round(average(rouge_l_lst) * 100,2)
    return results

import rouge

def prepare_results(p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)

def calculate_rouge155(all_modelsums, all_references):
    apply_avg = 'Avg'
        #apply_best = aggregator == 'Best'

    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                               max_n=4,
                               limit_length=True,
                               length_limit=100,
                               length_limit_type='words',
                               apply_avg=apply_avg,
                               apply_best=False,
                               alpha=0.5, # Default F1_score
                               weight_factor=1.2,
                               stemming=True)

    scores = evaluator.get_scores(all_modelsums, all_references)
    #print(scores)
    
    rouge_1 = 0
    rouge_2 = 0
    rouge_l = 0
    for metric, results in sorted(scores.items()):
        #print(prepare_results(results['p'], results['r'], results['f']))
        if metric == 'rouge-1':
            rouge_1 = results['f']
        if metric == 'rouge-2':
            rouge_2 = results['f']
        if metric == 'rouge-l':
            rouge_l = results['f']
    #print(rouge_1 + rouge_2 + rouge_l)
    return rouge_1, rouge_2, rouge_l
