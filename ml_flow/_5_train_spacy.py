import spacy
from spacy.util import minibatch, compounding
import torch
from sklearn.metrics import f1_score, accuracy_score, classification_report
import time
import random
import pandas as pd

def train_spacy(label_values, train_data, valid_texts, valid_labels, iterations, model_arch, dropout, learn_rate, output_dir):

	"""
	This function creates a spacy model which trains on the train_data provided and validates results with valid data
	printing the classification report 

	label_values: list of unique label values

	train_data: training data in the spacy format

	valid_texts: list of validation text

	valid_labels: list of validation labels

	iterations: number of iterations to train the training data on the model

	model_arch: model architeture string, either 'bow', 'simple_cnn' or 'ensemble'

	dropout: dropout rate

	learn_rate: learning rate

	output_dir: path to save the trained model 

	return: nlp
	"""

	nlp = spacy.load('en_core_web_lg')

	print('\nModel loading complete')

	textcat = nlp.create_pipe('textcat', config={'exclusive_classes':True, 'architecture': model_arch})
	nlp.add_pipe(textcat)

	for _, label in enumerate(label_values):
		textcat.add_label(label)

	pipe_exceptions = ['textcat']
	other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

	with nlp.disable_pipes(*other_pipes):
		optimizer = nlp.begin_training()
		optimizer.learn_rate = learn_rate
		print('\nTraining the model..')
		total_start_time = time.clock()

	for i in range(iterations):
		print('\nIteration:', str(i+1))
		start_time = time.clock()
		losses = {}
		true_labels = []
		pred_labels = []

		random.shuffle(train_data)
		batches = minibatch(train_data, size=compounding(4., 32., 1.001))
		for batch in batches:
			texts, annotations = zip(*batch)
			nlp.update(texts, annotations, sgd=optimizer, drop=dropout, losses=losses)

		with textcat.model.use_params(optimizer.averages):
			nlp.to_disk(output_dir)

			docs = [nlp.tokenizer(text) for text in valid_texts]

			for j, doc in enumerate(textcat.pipe(docs)):
				true_series = pd.Series(valid_labels[j]['cats'])
				true_label = true_series.idxmax()
				true_labels.append(true_label)

				pred_series = pd.Series(doc.cats)
				pred_label = pred_series.idxmax()
				pred_labels.append(pred_label)

			score_f1 = f1_score(true_labels, pred_labels, average='weighted')
			score_ac = accuracy_score(true_labels, pred_labels)

			print(classification_report(true_labels, pred_labels))
			print('\ntTextcat_loss: {:.3f}\t f1_score: {:.3f}\t accuracy_score: {:.3f}'.format(losses['textcat'], score_f1, score_ac))

			print('\nElapsed time:', str(round((time.clock() - start_time)/60,2)) + ' minutes')

	print('\nTraining complete')
	print('\nTotal time:', str(round((time.clock() - total_start_time)/60,2)) + ' minutes')

	return nlp




