import pandas as pd
from fastai.text import *
import sklearn.feature_extraction.text as sklearn_text
import pickle
from baseline_model import BaselineModel

datapath = 'data'
path = Path(datapath)
df = pd.read_csv(path/'train.csv')
df["is_humor"] = df.is_humor.map({1:'yes', 0:'no'})


#select task as label. Here, the humor classification task has been chosen

df = pd.DataFrame({
    'label':df['is_humor'],
    'text': df['text'].replace(r'\n', ' ', regex=True),
    
})

df.to_csv(datapath+'/train1.csv', index=False)


def create_label(regression:bool = False):
    count = 0
    error = True
    while error:
      try: 
        # Preprocessing steps
        if regression:
          rating = (TextList.from_csv(path, 'train1.csv', cols='text')
                         .split_by_rand_pct()
                         .label_from_df(cols=0, label_cls=FloatList))
          error = False
          print(f'failure count is {count}\n')
        else:

          rating = (TextList.from_csv(path, 'train1.csv', cols='text')
                         .split_by_rand_pct()
                         .label_from_df(cols=0))
          error = False
          print(f'failure count is {count}\n')      
      except: # catch *all* exceptions
            # accumulate failure count
        count = count + 1
        print(f'failure count is {count}')
    return rating
        
  
rating = create_label()

model = BaselineModel(rating)
model.train()


