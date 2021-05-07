from fastai.text import *
import sklearn.feature_extraction.text as sklearn_text
import pickle
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from dataclasses import asdict, dataclass, field, fields
from sklearn.metrics import mean_squared_error

class BaselineModel:

  def __init__(self, rating: LabelLists=None, regression:bool = False):
    self.rating = rating
    self.regression = regression;
    vectorizer = CountVectorizer(ngram_range=(1,3), preprocessor=noop, tokenizer=noop, max_features=800000)
    train_docs = self.rating.train.x
    train_words = [[self.rating.vocab.itos[o] for o in doc.data] for doc in train_docs]
    valid_docs = self.rating.valid.x
    valid_words = [[self.rating.vocab.itos[o] for o in doc.data] for doc in valid_docs]
    self.train_veczr = vectorizer.fit_transform(train_words)
    self.valid_veczr = vectorizer.transform(valid_words)
    self.vocab = vectorizer.get_feature_names()
  
  
  def train(self):
    y=self.rating.train.y
    if self.regression:
     # fit model
     m = LinearRegression()
     m.fit(self.train_veczr.sign(), y.items);
     # get predictions
     preds = m.predict(self.valid_veczr.sign())
     error = mean_squared_error(self.rating.valid.y.items, preds, squared=False)
     print("RMSE: ", error)
    else:
     # fit model
     yes = y.c2i['yes']
     no = y.c2i['no'] 
     m = LogisticRegression(C=0.1, dual=False, solver = 'liblinear')
     m.fit(self.train_veczr.sign(), y.items);
    
     # get predictions
     preds = m.predict(self.valid_veczr.sign())
     valid_labels = [label == yes for label in self.rating.valid.y.items]
     # check accuracy
     accuracy = (preds==valid_labels).mean()
     print(f'Accuracy = {accuracy} for Logistic Regression, with binarized trigram counts from `CountVectorizer`' )

    
