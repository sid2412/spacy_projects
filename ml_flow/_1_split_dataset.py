import pandas as pd 
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def split_dataset(data_path, 
				text_col='consumer_complaint_narrative', 
				cat_col='product', 
				n_rows=None, 
				test_size=0.15, 
				valid_set=True, 
				valid_size=0.15):

	"""
	This function splits a dataframe in csv format to train/test/valid sets

	data_path: path to the data in csv format

    text_col: the text column that will be trained for classification, defaults to 'consumer_complaint_narrative'

    cat_col: the categories in which the text will be classified to, defaults to 'product'

    n_rows: Number of rows to be used for training. If None, selects entire dataset

    test_size: test set size, accepts float between 0.0 and 1.0

    valid_set: Set False if valid set is not required, defaults to True

    valid_size: valid set size, accepts float between 0.0 and 1.0

    :return: train_X, train_y, test_X, test_y, valid_X, valid_y
	"""

	print('\nReading data..')

	df = pd.read_csv(data_path)

	df.dropna(subset=[text_col], axis=0, inplace=True)

	df = df[[text_col, cat_col]]

	df = shuffle(df)

	all_rows = len(df)
	if n_rows is not None:
		df = df[[cat_col, text_col]][:n_rows]
		print('\nSelected {} rows of dataset'.format(n_rows))
	else:
		df = df[[cat_col, text_col]][:all_rows]
		print('\nSelected all rows i.e. {} rows of dataset'.format(all_rows))

	df.reset_index(inplace=True)

	print('\nSplitting dataset in train and test sets..')

	train_X, test_X, train_y, test_y = train_test_split(df[text_col],
														df[cat_col],
														test_size=test_size,
														stratify=df[cat_col],
														random_state=36)



	if valid_set==True:
		train_X, valid_X, train_y, valid_y = train_test_split(train_X,
															train_y,
															test_size=valid_size,
															stratify=train_y,
															random_state=36)

		print('\nShape of train_X:', train_X.shape)
		print('Shape of train_y:', train_y.shape)

		print('\nShape of test_X:', test_X.shape)
		print('Shape of test_y:', test_y.shape)

		print('\nShape of valid_X:', valid_X.shape)
		print('Shape of valid_y:', valid_y.shape)

		print('\nText split complete for preprocessing')

		return train_X, train_y, test_X, test_y, valid_X, valid_y


	else:
		print('\nShape of train_X:', train_X.shape)
		print('Shape of train_y:', train_y.shape)

		print('\nShape of test_X:', test_X.shape)
		print('Shape of test_y:', test_y.shape)

		print('\nText split complete for preprocessing')

		return train_X, train_y, test_X, test_y

	





