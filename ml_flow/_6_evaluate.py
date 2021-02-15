import spacy
import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np

def evaluate(test_texts, test_labels, model, train_label_values):

	"""
	This function evaluates results on unseen test data based on the model saved weights

	test_texts: list of texts

	test_labels: list of labels

	model: saved trained model which will be loaded and used to make predictions

	train_label_values: list of unique label names
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

	def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
		
		"""
		This function prints and plots the confusion matrix.
		Normalization can be applied by setting `normalize=True`.
		"""

		if normalize:
			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
			print("Normalized confusion matrix")
		else:
			print('Confusion matrix, without normalization')

		# print(cm)

		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation=45)
		plt.yticks(tick_marks, classes)

		fmt = '.2f' if normalize else 'd'
		thresh = cm.max() / 2.
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			plt.text(j, i, format(cm[i, j], fmt), 
					horizontalalignment="center",
					color="white" if cm[i, j] > thresh else "black")
	
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.tight_layout()

	cnf_matrix = confusion_matrix(true_labels, pred_labels, labels=train_label_values)
	
	plt.figure(figsize=(10,8))
	plt.tight_layout()
	plot_confusion_matrix(cnf_matrix, classes=train_label_values, normalize=True)

	print(classification_report(true_labels, pred_labels))