import torch.nn as nn
import torch.nn.functional as F

class block(nn.Module):
    def __init__(self , n_hidden , bias , drop_out):
        super().__init__()
        self.ln = nn.LayerNorm(n_hidden , bias= bias)
        self.fc = nn.Linear(n_hidden , n_hidden, bias= bias)
        self.relu = nn.ReLU()
        self.dropout=nn.Dropout(drop_out)
        
    def forward(self,x):
        return self.dropout(self.relu(self.fc(self.ln(x))))
    
class DeepFakeDetectionModel(nn.Module):
    def __init__(self ,n_layers , n_hidden=128,bias = True , drop_out=0.1):
        super().__init__()
        self.ST_ln   = nn.LayerNorm(26 , bias = bias)
        self.ST_fc   = nn.Linear(26,n_hidden , bias = bias)
        self.dropout = nn.Dropout(drop_out)
        
        self.layers = nn.ModuleList([block(n_hidden,bias , drop_out) for _ in range(n_layers)])
        
        self.LAST_fc = nn.Linear(n_hidden , 1)
        self.LAST_ln = nn.LayerNorm(n_hidden)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = F.relu(self.ST_fc(self.ST_ln(x)))
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.LAST_ln(x)
        x = self.LAST_fc(x)
        x = self.sigmoid(x)
        return x
        
        
        