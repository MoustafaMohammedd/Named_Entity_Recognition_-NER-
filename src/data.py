import os
import sys  
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader ,Dataset
from sklearn.model_selection import train_test_split
from torchtext.vocab import build_vocab_from_iterator
from src.utils import pad_sentence ,prepared_tag,sentence_builder,pad_labels
from config.config import config_hp
from transformers import AutoTokenizer

bert_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df=pd.read_csv(r"D:\Named_Entity_Recognition_(NER)\name-entity-recognition-ner-dataset\NER dataset.csv",encoding='unicode_escape')
df["Tag"]=df["Tag"].apply(prepared_tag)

labels ={l:i for i,l in enumerate(df["Tag"].unique())}

def enc_labels (w):
    for l,i in labels .items():
        if l==w:
            return i
           
          
df["enc_labels"] =df["Tag"].apply(enc_labels)

df=df.ffill()

sentences = df.groupby("Sentence #").apply(sentence_builder, include_groups=False).tolist()

all_s=[]
all_l=[]
for ss in sentences:
    s=[]
    t=[]
    for w,l in ss :
        t.append(l)
        s.append(w)
    all_s.append(s)
    all_l.append(t)

our_vocab=build_vocab_from_iterator(all_s,specials=config_hp["SPECIAL_TOKENS"],max_tokens=config_hp["MAX_VOCAB"])
our_vocab.set_default_index(our_vocab[config_hp["SPECIAL_TOKENS"][1]])

df_data=pd.DataFrame({"tokens_s":all_s,"labels_enc":all_l})


x_train,x_test,y_train,y_test=train_test_split(df_data["tokens_s"],df_data["labels_enc"],test_size=0.2,shuffle=True,random_state=42)




class OurDataSet(Dataset):
  def __init__(self,x,y,vocab,max_len):
    self.x=x
    self.y=y
    self.vocab=vocab
    self.max_len=max_len

  def __len__(self):
    return len(self.x)
  def __getitem__(self, index):

    padded_s=pad_sentence(self.x.values[index],self.max_len,config_hp["SPECIAL_TOKENS"][0])
    padded_l=pad_labels(self.max_len,self.y.values[index])
    input_seq=torch.tensor(self.vocab.lookup_indices(padded_s), dtype=torch.long)
    target=torch.tensor(padded_l , dtype=torch.long)

    return input_seq ,target


our_train_data_set=OurDataSet(x=x_train,y=y_train,vocab=our_vocab,max_len=config_hp["MAX_LEN"])
our_test_data_set=OurDataSet(x=x_test,y=y_test,vocab=our_vocab,max_len=config_hp["MAX_LEN"])

def get_datasets_and_loaders_for_lstm ():
   our_train_data_loader=DataLoader(our_train_data_set,batch_size=config_hp["BATCH_SIZE"],shuffle=True)
   our_test_data_loader=DataLoader(our_test_data_set,batch_size=config_hp["BATCH_SIZE"],shuffle=False)
   return our_train_data_loader,our_test_data_loader,our_train_data_set,our_test_data_set


class BertData(Dataset):

    def __init__(self,s,l):
        self.s = s
        self.l=l
        


    def __len__(self):
        return len(self.s)

    def __getitem__(self,i):

        text=self.s[i]
        tokenized_input = bert_tokenizer(text, is_split_into_words=True, truncation=True, padding='max_length', max_length=100, return_tensors="pt")
        input=tokenized_input

        y=self.align_labels_with_tokens(self.l[i],input)

        target=torch.tensor(y, dtype=torch.long)

        return input,target



    def align_labels_with_tokens(self, labels, tokenized_input):
      aligned_labels = []
      word_ids = tokenized_input.word_ids()

      previous_word_id = None
      for word_id in word_ids:
          if word_id is None:
              aligned_labels.append(-100)
          elif word_id != previous_word_id:
              aligned_labels.append(labels[word_id])
          else:
              aligned_labels.append(labels[word_id])
          previous_word_id = word_id

      #aligned_labels=pad_t(100,aligned_labels)

      return aligned_labels

  
our_train_data_set_bert=BertData(x_train,y_train)
our_test_data_set_bert=BertData(x_test,y_test)

def get_datasets_and_loaders_for_bert ():
   our_train_data_loader_bert=DataLoader(our_train_data_set_bert,batch_size=config_hp["BATCH_SIZE"],shuffle=True)
   our_test_data_loader_bert=DataLoader(our_test_data_set_bert,batch_size=config_hp["BATCH_SIZE"],shuffle=False)
   return our_train_data_loader_bert,our_test_data_loader_bert,our_train_data_set_bert,our_test_data_set_bert




# if __name__ == "__main__":
#     for x,y in our_train_data_loader:
#         print(x.shape)
#         print(y.shape)
#         break

#     for x,y in our_test_data_loader:
#         print(x.shape)
#         print(y.shape)
#         break

