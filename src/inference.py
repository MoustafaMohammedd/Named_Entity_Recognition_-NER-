from models import OurLSTM, BertModel
from data import get_datasets_and_loaders_for_lstm,get_datasets_and_loaders_for_bert,labels
from config.config import config_hp
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from utils import our_predict_lstm
from transformers import AutoModel ,AutoTokenizer


our_train_data_loader, our_test_data_loader,our_train_data_set,our_test_data_set=get_datasets_and_loaders_for_lstm()
our_train_data_loader_bert,our_test_data_loader_bert,our_train_data_set_bert,our_test_data_set_bert=get_datasets_and_loaders_for_bert()

device="cuda" if torch.cuda.is_available() else "cpu"


our_lstm_model=OurLSTM(config_hp["MAX_VOCAB"],config_hp["EMBEDDING_SIZE"],config_hp["LSTM_HIDDEN_SIZE"],config_hp["LSTM_N_LAYERS"]).to(device)
checkpoint = torch.load(r"D:\Named_Entity_Recognition_(NER)\best_model_lstm\final_model.pth")
our_lstm_model.load_state_dict(checkpoint['model_state_dict'])

bert_model = AutoModel.from_pretrained("google-bert/bert-base-uncased")
for p in bert_model.parameters():
    p.requires_grad = False

our_bert_model=BertModel(bert_model).to(device)
checkpoint_bert = torch.load(r"D:\Named_Entity_Recognition_(NER)\best_model_bert\best_model.pth")
our_bert_model.load_state_dict(checkpoint_bert['model_state_dict'])

def evaluation_test_lstm():
    
    our_lstm_model.eval()
    with torch.inference_mode(): 
    
        pred_l=[]
        true_l=[]
        for x_test_batch , y_test_batch in our_test_data_loader : 
            x_test_batch=x_test_batch.to(device)
            y_test_batch=y_test_batch.to(device)
        
            y_test_pred=our_lstm_model(x_test_batch)
    
            
            y_test_pred=torch.argmax(torch.softmax(y_test_pred,dim=-1),dim=-1).squeeze().flatten().cpu().detach().numpy()
    
            
            y_test_batch=y_test_batch.flatten().cpu().detach().numpy()
    
            pred_l.extend(y_test_pred)
            true_l.extend(y_test_batch)
    
    pred_f=[]
    true_f=[]
    for true_label, pred_label in zip(true_l, pred_l):
        if true_label != -100:
            true_f.append(true_label)
            pred_f.append(pred_label)

    return true_f,pred_f
    

    

def evaluation_test_bert():
    
    our_bert_model.eval()
    with torch.inference_mode(): 
    
        pred_l=[]
        true_l=[]
        for x_test_batch , y_test_batch in our_test_data_loader_bert : 
            x_test_batch=x_test_batch.to(device)
            y_test_batch=y_test_batch.to(device)
        
            y_test_pred=our_bert_model(x_test_batch['input_ids'].squeeze(1),x_test_batch['attention_mask'].squeeze(1)).squeeze(1)
    
            
            y_test_pred=torch.argmax(torch.softmax(y_test_pred,dim=-1),dim=-1).squeeze().flatten().cpu().detach().numpy()
    
            
            y_test_batch=y_test_batch.flatten().cpu().detach().numpy()
    
            pred_l.extend(y_test_pred)
            true_l.extend(y_test_batch)
    
    pred_f=[]
    true_f=[]
    for true_label, pred_label in zip(true_l, pred_l):
        if true_label != -100:
            true_f.append(true_label)
            pred_f.append(pred_label)

    return true_f,pred_f

   
    
    
if __name__=="__main__":
    predictions, true_labels = evaluation_test_lstm()

    cm=confusion_matrix(true_labels,predictions) 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(r"D:\Named_Entity_Recognition_(NER)\images\confusion_matrix_lstm.png")
    plt.tight_layout()
    plt.show()
    
    print(our_predict_lstm("hello my Name is Alex and i am from France and Egypt", our_lstm_model, our_train_data_set.vocab, config_hp, labels,device))
    
    predictions, true_labels = evaluation_test_bert()

    cm=confusion_matrix(true_labels,predictions) 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(r"D:\Named_Entity_Recognition_(NER)\images\confusion_matrix_bert.png")
    plt.tight_layout()
    plt.show()
    print(our_predict_lstm("hello my Name is Alex and i am from France and Egypt", our_lstm_model, labels,device))
  
