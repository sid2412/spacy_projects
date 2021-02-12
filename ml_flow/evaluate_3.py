import spacy
import torch
import pandas as pd
from sklearn.metrics import classification_report

def evaluate(test_texts, test_labels, model):

	"""
	This function evaluates results on unseen test data based on the model saved weights

	test_texts: list of texts

	test_labels: list of labels

	model: saved trained model which will be loaded and used to make predictions
	"""

	nlp = spacy.load(model)

	print('\nModel loading complete')

	docs = [nlp.tokenizer(text) for text in test_texts]

	textcat = nlp.get_pipe('textcat')

	true_labels = []
	pred_labels = []

	for j, doc in enumerate(textcat.pipe(docs)):
		true_series = pd.Series(test_labels[j]['cats'])
		true_label = true_series.idxmax()
		true_labels.append(true_label)

		pred_series = pd.Series(doc.cats)
		pred_label = pred_series.idxmax()
		pred_labels.append(pred_label)

	print(classification_report(true_labels, pred_labels))