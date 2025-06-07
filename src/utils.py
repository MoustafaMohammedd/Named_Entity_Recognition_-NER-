import copy
import matplotlib.pyplot as plt
import torch
from torchtext.data import get_tokenizer

def prepared_tag(t):
    if t in ["B-art","B-eve","I-art","I-eve","B-nat","I-gpe","I-nat"]:
        return "O"
    else:
        return t
      
def sentence_builder(x):
    iterator = zip(x["Word"].values.tolist(),
                   x["enc_labels"].values.tolist())
    return [(word, tag) for word, tag in iterator]
  
  
tokenizer = get_tokenizer("basic_english")

def prepare_data (sentence):
  tokens=tokenizer(sentence)
  tokens_l=[]
  for token in tokens :
    if token.isalnum()== False :
      continue
    tokens_l.append(token.lower())

  return tokens_l


def pad_sentence (sentence_l,max_len,pad_token):
  if len(sentence_l)>=max_len:
    return sentence_l[:max_len]
  else:
    return sentence_l + [pad_token] * (max_len-len(sentence_l))

def pad_labels(max_len,l_labels):
    if max_len >=len(l_labels):
        return l_labels + [-100] * (max_len-len(l_labels))
    else:
        return l_labels[:max_len]


def our_accuracy(t,y): 
  t=t.flatten()  #32*60  >>1920
  y=torch.argmax(torch.softmax(y,dim=-1),dim=-1).squeeze().flatten() #32*60*10 >> 32*60*1 >> 32*60 >>1920
  mask = (t != -100)
  correct = (t==y)*mask
  correct_count = correct.sum().item()
  total_count = mask.sum().item()
  return correct_count , total_count   

# [1,1,1,1,0,0,0,-100,-100]  [1,1,1,1,1,1,1,0,0]  [1,0,1,1,1,0,1,-100,-100]



class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.best_accuracy = None  # New attribute to track best accuracy
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss, val_accuracy):
        if self.best_loss is None or val_loss < self.best_loss:  # Update if val_loss is better
            self.best_loss = val_loss
            self.best_accuracy = val_accuracy  # Update best accuracy along with best loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}. " \
                          # f"Best Loss: {self.best_loss:.4f}, Best Accuracy: {self.best_accuracy:.2f}%"

        elif self.best_loss - val_loss >= self.min_delta:
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}. " \
                          # f"Best Loss: {self.best_loss:.4f}, Best Accuracy: {self.best_accuracy:.2f}%"

        else:
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs. " \
                          # f"Best Loss: {self.best_loss:.4f}, Best Accuracy: {self.best_accuracy:.2f}%"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs. " \
                              f"Best Loss: {self.best_loss:.4f}, Best Accuracy: {self.best_accuracy:.2f}%"
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False


def save_checkpoint(model, optimizer, epoch, loss, filename="model_checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filename)


def plot_l_a(train_loss,test_loss,train_accuracy,test_accuracy,name): 
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    
    axs[0].plot(train_loss, label='Training Loss')
    axs[0].plot(test_loss, label='Test Loss')
    axs[0].set_title('Training and Test Loss over Epochs')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    #axs[0].set_ylim([0, 1])
    axs[0].legend()
    
    axs[1].plot(train_accuracy, label='Training Accuracy')
    axs[1].plot(test_accuracy, label='Test Accuracy')
    axs[1].set_title('Training and Test Accuracy over Epochs')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    #axs[1].set_ylim([0, 100])
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig (r'D:\Named_Entity_Recognition_(NER)\images' + f"\{name}")
    plt.show()
    
    

  
def our_predict_lstm(s,our_model, our_vocab,config_hp,labels, device):
  
    len_s=len(s.split())
    s=pad_sentence(s.split(),config_hp["MAX_LEN"],config_hp["SPECIAL_TOKENS"][0])
    s=torch.tensor(our_vocab.lookup_indices(s), dtype=torch.long).unsqueeze(0).to(device) 
    out=our_model(s)
    out=torch.argmax(torch.softmax(out,dim=-1),dim=-1).squeeze()
    out=out.cpu().detach().numpy()
    f_out=[]
    for i in out: 
        for k,v in labels.items(): 
            if i==v :
                f_out.append(k)
                break
    
    return f_out[:len_s]
  
  
 
def our_predict_bert(s,our_model, labels, device):
  
    len_s=len(s.split())
    
    tokenized_input = tokenizer(s.split(), is_split_into_words=True, truncation=True, padding='max_length', max_length=100, return_tensors="pt").to(device)

    out=our_model(tokenized_input["input_ids"].squeeze(1) ,tokenized_input["attention_mask"].squeeze(1) ).squeeze(1)
    out=torch.argmax(torch.softmax(out,dim=-1),dim=-1).squeeze()
    out=out.cpu().detach().numpy()
    f_out=[]
    for i in out: 
        for k,v in labels.items(): 
            if i==v :
                f_out.append(k)
                break
    l=tokenized_input["attention_mask"][0].sum().item()
    return f_out[:l]