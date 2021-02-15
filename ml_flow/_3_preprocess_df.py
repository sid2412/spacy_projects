import pandas as pd
import re
import string

from nltk.corpus import stopwords
stop = stopwords.words('english')

from sklearn.model_selection import train_test_split

def preprocess_df(data_path, text_col='consumer_complaint_narrative', cat_col='product', n_rows=None):

    """
    This function reads the data, cleans it and preprocesses it to spacy format for training.

    data_path: path to the data in csv format
    
    text_col: the text column that will be trained for classification, defaults to 'consumer_complaint_narrative'
    
    cat_col: the categories in which the text will be classified to, defaults to 'product'
    
    n_rows: Number of rows to be used for training. If None, selects entire dataset
    
    return: label_values, train_data, valid_texts, valid_labels, test_texts, test_labels
    """

    print('\nReading data..')

    df = pd.read_csv(data_path)

    df.dropna(subset=[text_col], axis=0, inplace=True)

    df = df[[text_col, cat_col]]

    print('\nCleaning text..')

    def clean_text(text):
        text = text.lower()
        text = re.sub('\{.*?\}', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = re.sub('\n', '', text)
        return text

    df['clean_text'] = df[text_col].apply(clean_text)

    def remove_xx(text):
        words = text.split()
        for word in words:
            if len(word) >= 2:
                if word[0] == 'x' and word[1] == 'x':
                    words.remove(word)
        return ' '.join(words)

    df['clean_text'] = df['clean_text'].map(lambda x: remove_xx(x))

    df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))

    print('\nText cleaning complete')

    all_rows = len(df)
    if n_rows is not None:
        df = df[[cat_col, 'clean_text']][:n_rows]
        print('\nSelected {} rows for training'.format(n_rows))
    else:
        df = df[[cat_col, 'clean_text']][:all_rows]
        print('\nSelected all rows i.e. {} rows for training'.format(all_rows))
    
    df.reset_index()

    label_values = list(df[cat_col].unique())

    print('\nConverting dataframe into series..')

    train_X = df['clean_text'].squeeze()
    train_y = df[cat_col].squeeze()

    print('\nShape of train_X:', train_X.shape)
    print('Shape of train_y:', train_y.shape)

    print('\nConverting data to spacy format')
    
    train_y_df = pd.get_dummies(train_y)

    train_texts = train_X.tolist()
    train_cats = train_y_df.to_dict(orient='records')

    train_data = list(zip(train_texts, [{'cats': cats} for cats in train_cats]))

    train_texts, train_labels = list(zip(*train_data))

    print('\nData is now ready to be trained')

    return label_values, train_data, train_texts, train_labels
