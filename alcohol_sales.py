import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

df = pd.read_csv('Alcohol_sales.csv', index_col=None, parse_dates=True)

#preproccessing data
y = df['S4248SM144NCEN'].values.astype(float)

test_size = 12

#splitting data for evaluation
training_set = y[:-test_size]
test_set = y[-test_size:]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
train_norm = scaler.fit_transform(training_set.reshape(-1, 1))

window_size = 12

train_norm = torch.FloatTensor(train_norm).view(-1)

#converting data into sequences for the RNN model
def input_data(seq, ws):
    out = []
    l = len(seq)
    
    for i in range(l-ws):
        window = seq[i:i+ws]
        label = seq[i+ws:i+ws+1]
        out.append((window, label))
        
    return out

train_data = input_data(train_norm, window_size)

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=200, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size)
        
        self.linear = nn.Linear(hidden_size, output_size)
        
        self.hidden = (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))
    
    def forward(self, seq):
        lstm_out, self.hidden = self.lstm(seq.view(len(seq), 1, -1), self.hidden)
        pred = self.linear(lstm_out.view(len(seq), -1))
        return pred[-1]

GPU = torch.device('cuda:0')

torch.manual_seed(101)
model = LSTM().to(GPU)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100

#training data
for i in tqdm(range(epochs)):
    for seq, y_train in train_data:
        optimizer.zero_grad()
        model.hidden = (torch.zeros(1, 1, model.hidden_size).to(GPU), 
                        torch.zeros(1, 1, model.hidden_size).to(GPU))
        y_pred = model(seq.to(GPU))
        loss = criterion(y_pred, y_train.to(GPU))
        loss.backward()
        optimizer.step()
    if i%20==0:
        print(f'epoch : {i}, loss : {loss}')


future = 12

preds = train_norm[-window_size:].tolist()

model.eval()
for i in range(future):
    seq = torch.FloatTensor(preds[-window_size:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_size).to(GPU), 
                        torch.zeros(1, 1, model.hidden_size).to(GPU))        
        preds.append(model(seq.to(GPU)).item())

    #predicting the future
true_preds = scaler.inverse_transform(np.array(preds[window_size:]).reshape(-1, 1))

x = np.arange('2018-02-01', '2019-02-01', dtype='datetime64[M]')

#visualizing the prediction with real data
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(df['S4248SM144NCEN']['2017-01-01':])
plt.plot(x, true_preds)
plt.show()














