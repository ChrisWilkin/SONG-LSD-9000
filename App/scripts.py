import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys, os
import Practice.transformer as trans
from torch.utils.data import DataLoader, Dataset
import csv
import torch.optim

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__)) #Absolute path of this file


class CustomDataset(Dataset):
    def __init__(self, cutoffs, labels):
        super().__init__()
        self.lyrics = cutoffs
        self.labels = labels

        print(type(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.lyrics[index], self.labels[index]

def cutoff(arr):
    try:
        arr = arr[1:-1]
        arr = np.array([row for row in csv.reader([arr], skipinitialspace=True, quotechar="'")]).squeeze()
        if len(arr) >= 20:
            arr = arr[:20]
        if len(arr) < 20:
            arr = np.append(arr, ['<PAD>' for _ in range(20 - len(arr))])
        arr = np.append(arr, '<EOS>')
        arr = np.insert(arr, 0, '<SOS>')
        return arr
    except:
        return ['<SOS>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', 
                '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', 
                '<PAD>', '<PAD>', '<PAD>', '<EOS>']

def sent_2_vec(sentence, vocab):
    return [vocab[word] for word in sentence]



file = pd.read_csv(os.path.join(THIS_FOLDER, 'Data\\data_tokens_stems.csv'))

df = pd.DataFrame(file)
df = df[['Stems', 'Positivity']]

df['Cutoffs'] = df['Stems'].apply(cutoff)

VOCAB = set(np.concatenate(df['Cutoffs'].to_numpy()))
VOCAB_list = list(VOCAB)
VOCABULARY = {keys:value for (value, keys) in enumerate(VOCAB_list)}


VOCAB_SIZE = len(VOCAB)
HEADS = 8
EMBED_SIZE = 256
NUM_LAYERS = 3
MAX_LENGTH = 22
DEVICE = 'cpu' if torch.cuda.is_available() else 'cpu'
EXPANSION = 4
DROP = 0.1
PAD = VOCABULARY['<PAD>']


LR = 3e-4
EPOCHS = 1
BATCH = 8

dset = CustomDataset(df['Cutoffs'].to_list(), torch.tensor(df['Positivity'].values))
dloader = DataLoader(dset, batch_size=BATCH, shuffle=True)

criterion = nn.MSELoss()

model = trans.Encoder(VOCAB_SIZE, EMBED_SIZE, NUM_LAYERS, HEADS, DEVICE, EXPANSION, DROP, MAX_LENGTH, PAD)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

sentence = ['<SOS>', 'i', 'am', 'super', 'mad', 'and', 'mad', 'to', 'be', 'mad', '!', 'i', 'am', 'super', 'mad', 'to', 'be', 'mad', 'mad', 'mad', 'mad', '<EOS>']
enc_sent = sent_2_vec(sentence, VOCABULARY)
print(enc_sent)

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1} / {EPOCHS}')
    model.train()

    #for idx, batch in enumerate(dloader):
    for idx in range(200):
        optimizer.zero_grad()
        inp_data, label = torch.tensor([enc_sent, enc_sent, enc_sent, enc_sent]),torch.tensor([0., 0., 0., 0.])

        output = model(inp_data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if (idx + 1) % 10 == 0:
            print(loss)
            print(output.shape)

        










   






