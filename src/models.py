import os 
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch 
import torch.nn as nn
from config.config import config_hp



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OurLSTM(nn.Module): 
  def __init__(self,max_vocab,embedding_size,hidden_size,n_layers):

    super(OurLSTM,self).__init__()

    self.max_vocab=max_vocab
    self.embedding_size=embedding_size
    self.hidden_size=hidden_size
    self.n_classes=config_hp["N_CLASSES"]
    self.n_layers=n_layers
    
    self.embed=nn.Embedding(self.max_vocab,self.embedding_size,0)
    self.lstm=nn.LSTM(100,self.hidden_size,num_layers=self.n_layers,batch_first=True,bidirectional=True)
    self.liner1=nn.Linear(self.hidden_size*self.n_layers,32)
    self.output=nn.Linear(32,self.n_classes+1) 
    self.drop_out=nn.Dropout(.25)
    self.relu=nn.ReLU()
    
   

  def forward (self,x):
    x=self.embed(x)
    out,_=self.lstm(x)
    x=self.liner1(out)
   
    x=self.relu(x)
    x=self.drop_out(x)
    x=self.output(x)

    return x

class BertModel(nn.Module):
    def __init__(self, bert):

        super(BertModel, self).__init__()

        self.bert = bert
        self.dropout = nn.Dropout(0.25)
        self.linear1 = nn.Linear(768, 384)
        self.linear2 = nn.Linear(384, 64)
        self.linear3 = nn.Linear(64, 11)
        self.relu = nn.ReLU()


    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(input_ids, attention_mask).last_hidden_state
        output = self.linear1(pooled_output)
        output = self.relu(output)
        output = self.dropout(output)

        output = self.linear2(output)
        output = self.relu(output)
        output = self.dropout(output)

        output = self.linear3(output)

        return output



if __name__=="__main__": 
    our_model=OurLSTM(config_hp["MAX_VOCAB"],config_hp["EMBEDDING_SIZE"],config_hp["LSTM_HIDDEN_SIZE"],config_hp["LSTM_N_LAYERS"])
    print(our_model(torch.randint(low=0, high=3000, size=(32, 100))).shape)
