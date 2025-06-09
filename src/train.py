import os 
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from config.config import config_hp
import torch
import torch.nn as nn
from src.models import OurLSTM, BertModel
from src.data import get_datasets_and_loaders_for_lstm,get_datasets_and_loaders_for_bert
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from src.utils import our_accuracy,EarlyStopping,save_checkpoint, plot_l_a
from transformers import AutoModel 


our_train_data_loader, our_test_data_loader,our_train_data_set,our_test_data_set=get_datasets_and_loaders_for_lstm()

our_train_data_loader_bert,our_test_data_loader_bert,our_train_data_set_bert,our_test_data_set_bert=get_datasets_and_loaders_for_bert()



device="cuda" if torch.cuda.is_available() else "cpu"

our_lstm_model=OurLSTM(config_hp["MAX_VOCAB"],config_hp["EMBEDDING_SIZE"],config_hp["LSTM_HIDDEN_SIZE"],config_hp["LSTM_N_LAYERS"]).to(device)


bert_model = AutoModel.from_pretrained("google-bert/bert-base-uncased")
for p in bert_model.parameters():
    p.requires_grad = False

our_bert_model=BertModel(bert_model).to(device)

early_stopping = EarlyStopping(patience=5, min_delta=0)


def our_train_lstm(epochs,our_model,our_accuracy):
  
    our_loss=nn.CrossEntropyLoss(ignore_index=-100)
    our_optimizer =torch.optim.AdamW(our_model.parameters(),lr=config_hp["LEARNING_RATE"])
  
    writer = SummaryWriter('runs\lstm')
    
    train_loss_l=[]
    test_loss_l=[]
    train_accuracy_l =[]
    test_accuracy_l =[]
    
    for epoch in range(epochs): 
    
    
      train_loss_v=0.0
      
      test_loss_v=0.0
      

      train_correct = 0.0
      train_total = 0.0
        
      test_correct = 0.0
      test_total = 0.0
      
      our_model.train()  
      for x_train_batch , y_train_batch in tqdm (our_train_data_loader,f"Epoch = {epoch}") : 
        
        x_train_batch=x_train_batch.to(device)
        y_train_batch=y_train_batch.to(device)
    
       
    
        y_train_pred=our_model(x_train_batch)
    
        train_loss=our_loss(y_train_pred.view(-1,11),y_train_batch.view(-1))
    
        our_optimizer.zero_grad()
        train_loss.backward()
        our_optimizer.step()
          
        train_loss_v +=train_loss.item()

        correct_train, total_train = our_accuracy(y_train_batch, y_train_pred)
        train_correct += correct_train
        train_total += total_train
        
      our_model.eval()
      with torch.inference_mode(): 
        
        for x_test_batch , y_test_batch in our_test_data_loader : 
          
            x_test_batch=x_test_batch.to(device)
            y_test_batch=y_test_batch.to(device)
    
            y_test_pred=our_model(x_test_batch)
    
            test_loss=our_loss(y_test_pred.view(-1,11),y_test_batch.view(-1))
    
            test_loss_v +=test_loss.item()

            correct_test, total_test = our_accuracy(y_test_batch, y_test_pred)
            test_correct += correct_test
            test_total += total_test

    
      train_acc = train_correct / train_total if train_total > 0 else 0
      test_acc = test_correct / test_total if test_total > 0 else 0 
    
      train_loss_l.append(train_loss_v/our_train_data_set.__len__())
      test_loss_l.append(test_loss_v/our_test_data_set.__len__())
    
      train_accuracy_l.append(train_acc)
      test_accuracy_l.append(test_acc)


      writer.add_scalar("Loss/Train", train_loss_l[-1], epoch)
      writer.add_scalar("Loss/Test", test_loss_l[-1], epoch)
      writer.add_scalar("Accuracy/Train", train_accuracy_l[-1], epoch)
      writer.add_scalar("Accuracy/Test", test_accuracy_l[-1], epoch)  
    

      print(f"at epoch = {epoch+1} || train loss = {train_loss_l[-1]:.3f} and test loss = {test_loss_l[-1]:.3f} || train accuracy = {train_accuracy_l[-1]:.3f} and test accuracy = {test_accuracy_l[-1]:.3f}")
      
      if (epoch + 1) % 2 == 0:
            save_checkpoint(our_model,our_optimizer,epoch + 1, test_loss_l[-1],r'D:\Named_Entity_Recognition_(NER)\best_model_lstm\model_checkpoint.pth')
            print(f"Checkpoint saved at epoch {epoch + 1}.")

      if early_stopping(our_model, test_loss_l[-1],test_accuracy_l[-1]):
            print(early_stopping.status)
            save_checkpoint(our_model,our_optimizer,epoch + 1, test_loss_l[-1], r'D:\Named_Entity_Recognition_(NER)\best_model_lstm\best_model.pth')
            print("Early stopping triggered!")
            writer.close()
            break 

    if early_stopping.counter < early_stopping.patience:
        save_checkpoint(our_model, our_optimizer, epoch+1, test_loss_l[-1], r'D:\Named_Entity_Recognition_(NER)\best_model_lstm\final_model.pth')
        print("Training completed without early stopping. Final model saved.")
    
    return train_loss_l,test_loss_l,train_accuracy_l,test_accuracy_l


def our_train_bert(epochs,our_model,our_accuracy):
  
    our_loss=nn.CrossEntropyLoss(ignore_index=-100)
    our_optimizer =torch.optim.AdamW(our_model.parameters(),lr=config_hp["LEARNING_RATE"])
  
    writer = SummaryWriter('runs\bert')
    
    train_loss_l=[]
    test_loss_l=[]
    train_accuracy_l =[]
    test_accuracy_l =[]
    
    for epoch in range(epochs): 
    
    
      train_loss_v=0.0
      
      test_loss_v=0.0
      

      train_correct = 0.0
      train_total = 0.0
        
      test_correct = 0.0
      test_total = 0.0
      
      our_model.train()  
      for x_train_batch , y_train_batch in tqdm (our_train_data_loader_bert,f"Epoch = {epoch}") : 
        
        x_train_batch=x_train_batch.to(device)
        y_train_batch=y_train_batch.to(device)
    
       
    
        y_train_pred=our_model(x_train_batch["input_ids"].squeeze(1) ,x_train_batch["attention_mask"].squeeze(1) ).squeeze(1)
    
        train_loss=our_loss(y_train_pred.view(-1,11),y_train_batch.view(-1))
    
        our_optimizer.zero_grad()
        train_loss.backward()
        our_optimizer.step()
          
        train_loss_v +=train_loss.item()

        correct_train, total_train = our_accuracy(y_train_batch, y_train_pred)
        train_correct += correct_train
        train_total += total_train
        
      our_model.eval()
      with torch.inference_mode(): 
        
        for x_test_batch , y_test_batch in our_test_data_loader_bert : 
          
            x_test_batch=x_test_batch.to(device)
            y_test_batch=y_test_batch.to(device)
    
            y_test_pred=our_model(x_test_batch["input_ids"].squeeze(1) ,x_test_batch["attention_mask"].squeeze(1) ).squeeze(1)
    
            test_loss=our_loss(y_test_pred.view(-1,11),y_test_batch.view(-1))
    
            test_loss_v +=test_loss.item()

            correct_test, total_test = our_accuracy(y_test_batch, y_test_pred)
            test_correct += correct_test
            test_total += total_test

    
      train_acc = train_correct / train_total if train_total > 0 else 0
      test_acc = test_correct / test_total if test_total > 0 else 0 
    
      train_loss_l.append(train_loss_v/our_train_data_set.__len__())
      test_loss_l.append(test_loss_v/our_test_data_set.__len__())
    
      train_accuracy_l.append(train_acc)
      test_accuracy_l.append(test_acc)


      writer.add_scalar("Loss/Train", train_loss_l[-1], epoch)
      writer.add_scalar("Loss/Test", test_loss_l[-1], epoch)
      writer.add_scalar("Accuracy/Train", train_accuracy_l[-1], epoch)
      writer.add_scalar("Accuracy/Test", test_accuracy_l[-1], epoch)  
    

      print(f"at epoch = {epoch+1} || train loss = {train_loss_l[-1]:.3f} and test loss = {test_loss_l[-1]:.3f} || train accuracy = {train_accuracy_l[-1]:.3f} and test accuracy = {test_accuracy_l[-1]:.3f}")
      
      if (epoch + 1) % 2 == 0:
            save_checkpoint(our_model,our_optimizer,epoch + 1, test_loss_l[-1],r'D:\Named_Entity_Recognition_(NER)\best_model_bert\model_checkpoint.pth')
            print(f"Checkpoint saved at epoch {epoch + 1}.")

      if early_stopping(our_model, test_loss_l[-1],test_accuracy_l[-1]):
            print(early_stopping.status)
            save_checkpoint(our_model,our_optimizer,epoch + 1, test_loss_l[-1], r'D:\Named_Entity_Recognition_(NER)\best_model_bert\best_model.pth')
            print("Early stopping triggered!")
            writer.close()
            break 

    if early_stopping.counter < early_stopping.patience:
        save_checkpoint(our_model, our_optimizer, epoch+1, test_loss_l[-1], r'D:\Named_Entity_Recognition_(NER)\best_model_bert\final_model.pth')
        print("Training completed without early stopping. Final model saved.")
        
    return train_loss_l,test_loss_l,train_accuracy_l,test_accuracy_l
  
  
 
if __name__ == "__main__":
    train_loss_l,test_loss_l,train_accuracy_l,test_accuracy_l=our_train_lstm(epochs=config_hp["EPOCHS"],our_model=our_lstm_model,our_accuracy=our_accuracy)

    plot_l_a(train_loss_l,test_loss_l,train_accuracy_l,test_accuracy_l,"lstm_plot_training_results.png")

