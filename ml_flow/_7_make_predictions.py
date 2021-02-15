import spacy
import torch

def make_predictions(model, text):

	"""
	This function takes a string as input and outputs the predicted category
	model: The path to model to be loaded
	text: String of text 
	"""

	nlp = spacy.load(model)

	doc = nlp(text)

	key = max(doc.cats, key=lambda x:doc.cats[x])

	print('\nThe model predicts:', key)


def comp_pred(model, test_texts, test_labels, i):

	"""
	This function can be applied to the test set to compare trule and predicted labels
	model: The path to the model to be loaded
	test_texts: Tuple of test texts
	test_labels: Tuple of test labels
	i: index number. Should be between 0 and len(test_texts)
	"""

	nlp = spacy.load(model)

	doc = nlp(test_texts[i])

	pred_label = max(doc.cats, key=lambda x:doc.cats[x])
	true_label = list(test_labels[i]['cats'].keys())[list(test_labels[i]['cats'].values()).index(1)]

	print('Text:\n', test_texts[i])

	print('\nThe true label is:', true_label)

	print('\nThe model predicts:', pred_label)
