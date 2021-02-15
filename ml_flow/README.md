Spacy text classification pipeline flow on the US_consumer_complaints dataset

the .ipynb files are different pipeline flows based on the objective:

[split_predict_flow.ipynb](https://github.com/sid2412/spacy_projects/blob/main/ml_flow/split_predict_flow.ipynb) is for when you want to split the dataset into train/test/valid sets, then prepocess, train, evaluate and predict

[split_updatelabels_predict_flow.ipynb](https://github.com/sid2412/spacy_projects/blob/main/ml_flow/split_updatelabels_predict_flow.ipynb) is for when the split leads to diffence in the number of labels between split sets. The update_labels function adds the missing labels in the test/valid set and then continues with the preprocessing, training, evaluation and predictions

preprocess_predict_flow.ipynb is for when you directly wish to preprocess the data file and generate predictions based on a pre-saved model
