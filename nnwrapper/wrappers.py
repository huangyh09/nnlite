# A wrapper function for pytorch NN model

import torch
import numpy as np
from tqdm import tqdm    
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

class NNWrapper():
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        
    def train(self, train_loader, verbose=False):
        self.model.train()
        train_loss = 0
        for data, label in train_loader:
            # data, label = data.cuda(), label.cuda()
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item() * data.size(0)
            
        train_loss = train_loss / len(train_loader.dataset)
        return train_loss
    
    def predict(self, test_loader):
        self.model.eval()
        y_outputs = []
        test_loss = 0
        with torch.no_grad():
            for data, label in test_loader:
                # data, label = data.cuda(), label.cuda()
                output = self.model(data)
                y_outputs.append(output)
                
                if label is not None:
                    loss = self.criterion(output, label)
                    test_loss += loss.item() * data.size(0)
        
        y_outputs = torch.cat(y_outputs)     
        test_loss = test_loss / len(test_loader.dataset)
        return y_outputs, test_loss
    
    def fit(self, train_loader, epoch, validation_loader=None, verbose=False):
        self.train_losses = np.zeros(epoch)
        self.valid_losses = np.zeros(epoch)
        for i in tqdm(range(epoch)):
            self.train_losses[i] = self.train(train_loader, verbose=verbose)
            
            if validation_loader is not None:
                _y_pred, self.valid_losses[i] = self.predict(validation_loader)
            
            if verbose:
                print('Loss - Epoch: %s \tTraining: %.6f \tValidation: %.6f\n'
                      %(i, self.train_losses[i], self.valid_losses[i]))
    
    def evaluate(self, y_scores, y_obs, mode='classification'):
        """Under development
        """
        y_preds  = torch.argmax(y_scores, 1).cpu().data.numpy()
        y_scores = y_scores[:, 1].cpu().data.numpy() 
        y_obs    = y_obs[:, 1].cpu().data.numpy()
        
        acc = np.sum(y_preds == y_obs) / len(y_preds)
        auc = roc_auc_score(y_obs, y_scores)
        confu = confusion_matrix(y_obs, y_preds)
        
        return acc, auc, confu
    