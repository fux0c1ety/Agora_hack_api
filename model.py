import sklearn
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import word_tokenize
import time

class TextNormalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        
    # some text cleaning functions
    def convert_to_lower(self, text):
        return text.lower()

    def remove_numbers(self, text):
        number_pattern = r'\d+'
        without_number = re.sub(pattern=number_pattern, repl=" ", string=text)
        return without_number

    def lemmatizing(self, text):
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text)
        return " ".join(list(map(lemmatizer.lemmatize, tokens)))

    def remove_punctuation(self, text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def remove_stopwords(self, text):
        removed = []
        stop_words = list(stopwords.words("russian"))
        tokens = word_tokenize(text)
        for i in range(len(tokens)):
            if tokens[i] not in stop_words:
                removed.append(tokens[i])
        return " ".join(removed)

    def remove_extra_white_spaces(self, text):
        single_char_pattern = r'\s+[a-zA-Z]\s+'
        without_sc = re.sub(pattern=single_char_pattern, repl=" ", string=text)
        return without_sc
    
    def text_joining(self, df):
        return df['name']
    
    def text_clearing(self, dataframe):
        dataframe['text'] = [""]*len(dataframe)
        for i in dataframe.index:
            dataframe['text'][i] = dataframe['name'][i]+" "+" ".join(dataframe['props'][i])
        dataframe['text'] = dataframe['text'].str.lower()
        #products_df['name'] = products_df['name'].apply(lambda x: remove_numbers(x))
        dataframe['text'] = dataframe['text'].apply(lambda x: self.remove_punctuation(x))
        dataframe['text'] = dataframe['text'].apply(lambda x: self.remove_stopwords(x))
        dataframe['text'] = dataframe['text'].apply(lambda x: self.remove_extra_white_spaces(x))
        dataframe['text'] = dataframe['text'].apply(lambda x: self.lemmatizing(x))
        return dataframe['text']
    
    def all_df_to_vec(self, dataframe):
        X_tf_wob = self.vectorizer.fit_transform(self.text_clearing(dataframe))
        X_tf_wob = X_tf_wob.toarray()
        return X_tf_wob

    def resp_to_vec(self, dataframe):
        X = self.vectorizer.transform(self.text_clearing(dataframe))
        return X.todense()

class Model:
    def __init__(self):
        self.preprocessor = TextNormalyzer()
        self.model = GaussianNB()

    def fit(self, json):
        df = pd.read_json(json)
        self.products_df = df[df.is_reference==False]
        references_df = df[df.is_reference][['product_id', 'name', 'props']]
        train_data = self.preprocessor.all_df_to_vec(self.products_df)
        self.products_df['ref_id'] = [0]*len(self.products_df)
        self.idslist = []
        for i, ref_id in enumerate(references_df.product_id):
            self.products_df['ref_id'][self.products_df.reference_id==ref_id] = i
            self.idslist.append(ref_id)
        self.model.fit(train_data, np.asarray(self.products_df['ref_id'])[...,np.newaxis])
        print('OK')

    def idfind(self, number):
        return self.idslist[number]

    def predict(self, json):
        df = pd.read_json(json)
        data = self.preprocessor.resp_to_vec(df)
        predict = self.model.predict(data)
        df['reference_id'] = list(map(self.idfind, predict))
        return df[['id', 'reference_id']].to_json()

