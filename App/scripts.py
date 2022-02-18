from tabnanny import verbose
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys, os
import Practice.transformer as trans
from torch.utils.data import DataLoader, Dataset
import csv
import torch.optim
import matplotlib.pyplot as plt

clear = lambda: os.system('cls')

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__)) #Absolute path of this file


class CustomDataset(Dataset):
    def __init__(self, cutoffs, labels, sent2vec, vocab):
        super().__init__()
        self.lyrics = cutoffs
        self.labels = labels
        self.lyrics = torch.tensor([sent2vec(sent, vocab) for sent in self.lyrics])

        print(type(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        lyrics = self.lyrics[index]
        scores = self.labels[index]
        return {'lyrics': lyrics, 'scores': scores}

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

clear()
print('Reading CSV File...')
file = pd.read_csv(os.path.join(THIS_FOLDER, 'Data\\data_tokens_stems.csv'))

df = pd.DataFrame(file)
df = df[['Stems', 'Positivity']]
clear()
print('Cutting Lyrics to 20 words...')
df['Cutoffs'] = df['Stems'].apply(cutoff)
clear()
print('Generating Vocabulary...')
VOCAB = set(np.concatenate(df['Cutoffs'].to_numpy()))
VOCAB_list = list(VOCAB)
VOCABULARY = {keys:value for (value, keys) in enumerate(VOCAB_list)}


VOCAB_SIZE = len(VOCAB)
HEADS = 8
EMBED_SIZE = 256
NUM_LAYERS = 3
MAX_LENGTH = 22
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EXPANSION = 4
DROP = 0.1
PAD = VOCABULARY['<PAD>']

LR = 3e-3
EPOCHS = 5
BATCH = 128
clear()
print('Creating Dataset and Dataloader...')
dset = CustomDataset(df['Cutoffs'].to_list(), torch.tensor(df['Positivity'].values), sent_2_vec, VOCABULARY)
dloader = DataLoader(dset, batch_size=BATCH, shuffle=True)

criterion = nn.BCELoss()

model = trans.Encoder(VOCAB_SIZE, EMBED_SIZE, NUM_LAYERS, HEADS, DEVICE, EXPANSION, DROP, MAX_LENGTH, PAD)
model.double()
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=512, gamma=0.2, verbose=True)

sentence = ['<SOS>', 'i', 'am', 'super', 'mad', 'and', 'mad', 'to', 'be', 'mad', '!', 'i', 'am', 'super', 'mad', 'to', 'be', 'mad', 'mad', 'mad', 'mad', '<EOS>']
enc_sent = sent_2_vec(sentence, VOCABULARY)
print(enc_sent)

losses = []
clear()
print('Training...')
print(f'Using {DEVICE}')
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1} / {EPOCHS}')
    model.train()

    for idx, batch in enumerate(dloader):
        optimizer.zero_grad()
        #inp_data, label = torch.tensor([enc_sent, enc_sent, enc_sent, enc_sent]),torch.tensor([0., 0., 0., 0.])
        inp_data, label = batch['lyrics'].to(DEVICE), batch['scores'].to(DEVICE)

        output = model(inp_data)
        loss = criterion(output.squeeze(), label)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        lr_scheduler.step()

        if (idx + 1) % 200 == 0:
            print(f'Batch {idx + 1} / {len(dloader)}')
            print(f'Loss: {np.average(losses[-199:]):.5f}')
            print(f'Prediction: {output[0]}; Actual: {label[0]}')
        

plt.plot(losses)
plt.show()
        










   






