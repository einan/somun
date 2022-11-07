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

directory = 'cnn/stories/'
stories = load_stories(directory)
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
