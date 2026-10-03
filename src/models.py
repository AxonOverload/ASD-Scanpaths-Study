import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence




def collate_fn(batch):

    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])

    padded_sequences = pad_sequence(sequences, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.float32)

    return padded_sequences, lengths, labels





class LSTM(nn.Module):

    def __init__(self,input_size=513,hidden_size=96):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size,num_layers=2, dropout=0.5, batch_first=True)
        
        self.fc = nn.Linear(hidden_size,1)



    def forward(self,padded_sequences,lengths):

        packed = pack_padded_sequence(padded_sequences,lengths.cpu(),
                                       batch_first=True,enforce_sorted=False)

        out,(h_n,c_n) = self.lstm(packed)

        last_hidden = h_n[-1]

        logits = self.fc(last_hidden)

        return logits.squeeze(-1)




class GRU(nn.Module):

    def __init__(self,input_size=513,hidden_size=64):
        super().__init__()

        self.gru = nn.GRU(input_size=input_size,hidden_size=hidden_size,
                          num_layers=2,dropout=0.5,batch_first=True)

        self.fc = nn.Linear(hidden_size,1)



    def forward(self,padded_sequences,lengths):

        packed = pack_padded_sequence(padded_sequences,lengths.cpu(),
                                      batch_first=True,enforce_sorted=False)

        packed_out, h_n = self.gru(packed)

        last_hidden = h_n[-1]

        logits = self.fc(last_hidden)

        return logits.squeeze(-1)
