import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import os
import time

'''
This takes the original dataset and creates a .csv file with 
Lyrics, Tokenized Lyrics, Stemmed+Tokenized Lyrics, Positivity Rating
data_tokens_stems.csv
'''

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__)) #Absolute path of this file

def remove_rn(x):
    '''
    Removed \r and \n from strings
    x: string
    reg: regex condition
    '''

    x = x.replace('\r', ' ')
    x = x.replace('\n', ' ')
    x = x.replace(',', '')
    x = x.replace("'", '')
    return x

file = pd.read_csv(os.path.join(THIS_FOLDER, '..\\Data\\data.csv'))
df = pd.DataFrame(file)

df = df[['seq','label']]
df.columns = ['Lyrics', 'Positivity']
df['Lyrics'] = [remove_rn(lyric) for lyric in df['Lyrics']]

#Stem and Tokenize
ps = PorterStemmer()

df['Tokens'] = df['Lyrics'].apply(word_tokenize)
df['Stems'] = df['Tokens'].apply(lambda x: [ps.stem(y.lower()) for y in x])

df.to_csv('data_tokens_stems.csv')








