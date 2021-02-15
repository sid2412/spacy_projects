import pandas as pd 
import string
import re
from nltk.corpus import stopwords
stop = stopwords.words('english')

def preprocess_series(train_X, train_y):

	"""
	This function takes a series as input, cleans it and converts to spacy format
	train_X: a series of input texts
	train_y: a series of labels
	returns: label_values, train_data, train_texts, train_labels
	"""

	print('\nCleaning text..')

	def clean_text(text):
		text = text.lower()
		text = re.sub('\{.*?\}', '', text)
		text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
		text = re.sub('\w*\d\w*', '', text)
		text = re.sub('\n', '', text)
		return text

	def remove_xx(text):
		words = text.split()
		for word in words:
			if len(word) >= 2:
				if word[0] == 'x' and word[1] == 'x':
					words.remove(word)
		return ' '.join(words)

	train_X = train_X.apply(lambda x: clean_text(x)).apply(lambda x: remove_xx(x)).apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))

	print('\nText cleaning complete')

	label_values = list(train_y.unique())
	print('\nNumber of labels:', len(label_values))
	print(label_values)

	print('\nConverting data to spacy format')

	train_y_df = pd.get_dummies(train_y)

	train_texts = train_X.tolist()
	train_cats = train_y_df.to_dict(orient='records')

	train_data = list(zip(train_texts, [{'cats': cats} for cats in train_cats]))

	train_texts, train_labels = list(zip(*train_data))

	print('\nData is now ready to be trained')

	return label_values, train_data, train_texts, train_labels

