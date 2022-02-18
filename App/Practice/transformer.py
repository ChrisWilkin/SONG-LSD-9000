from wsgiref import headers
import torch
from torch import DeviceObjType, nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        '''
        heads - how many sections to split the embedding (embed_size) into before attention   
        '''
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size  
        self.heads = heads
        self.head_dim = embed_size // heads #integer division

        assert self.head_dim * heads == embed_size, 'Embed size needs to be divisible by heads'

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(heads * self.head_dim, embed_size) #val, key, queries will be concatentated, producing this input dim

    def forward(self, values, keys, query, mask):
        N = query.shape[0] #Number of queries being sent in
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        #Split embedding into heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        query = self.queries(query)

        energy = torch.einsum('nqhd,nkhd->nhqk', [query, keys])
        # query shape: [N, query_len, heads, head_dim] - ditto for keys and query
        #energy shape: (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20")) # Cut out values closed by mask and set to effectively negative infinity

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim = 3) # along key dimension

        out = torch.einsum('nhql,nlhd->nqhd', [attention, values]).reshape(
            N, query_len, self.heads*self.head_dim
        ) # key_len = value_len
        # attention shape: (N, heads, query_len, keay_len)
        # values shape: (N, value_len, heads, heads_dim)
        # after: (N, query_len, heads, lead_dim) then flatten last two dimensions

        out = self.fc_out(out)

        return out

    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size) # similar to batch norm, but this normalises every example rather than every batch
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))

        return out


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length, pad_idx):
        #max_length related to position embedding - tell it how long the longest sentence length is - allows larger onces to be ignored
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.src_pad_idx = pad_idx
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size) # created embeddings for a vocabulary size x, with each word being assigned a vector size y
        self.positional_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size, heads, dropout=dropout, forward_expansion=forward_expansion
                ) for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_size, 1)
        self.fc_out2 = nn.Linear(22, 1)

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def forward(self, x, mask=None):
        #if mask is None:
         #   mask = self.make_src_mask(x)

        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.positional_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask) # in the encode, the value key and query are all the same!
        
        # Softmax and Linear Stage for positivity score
        out = torch.sigmoid(self.fc_out2(self.fc_out(out).squeeze()))

        return torch.mean(out, dim=1)


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()

        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)

        return out

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device) for  _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def foward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = X.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        x = self.dropout((self.word_embedding(x) + self.positional_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_dix, embed_size=256, num_layers=6, forward_expansion=4, heads=8, dropout=0, device='cuda', max_langth=100):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_langth)
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_langth)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_dix
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)

        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)

        return out

        

