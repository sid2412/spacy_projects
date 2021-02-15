def update_labels(train_label_values, test_label_values, test_data):

	"""
	This function is used when the test/valid sets have missing labels compared to the train set
	train_label_values: List of label names in train set
	test_label_values: List of label names in test set
	test_data: test data in the spacy format
	return: test_label_values, test_data, test_texts, test_labels
	"""

	missing_label = [label for label in train_label_values if label not in test_label_values]

	for j, data in enumerate(test_data):
		for i, _ in enumerate(missing_label):
			test_data[j][1]['cats'].update({missing_label[i]: 0})

	test_texts, test_labels = list(zip(*test_data))
	test_label_values = list(test_labels[0]['cats'].keys())

	print('Label values have been updated')
	print('\nNumber of labels:', len(test_label_values))
	print(test_label_values)

	return test_label_values, test_data, test_texts, test_labels