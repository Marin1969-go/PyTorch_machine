import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_breast_cancer
torch.manual_seed(1)
data_load = load_breast_cancer()

_global_var_data_x = torch.tensor(data_load.data, dtype=torch.float32)
_global_var_target = torch.tensor(data_load.target, dtype=torch.int64)

class BatchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(30, 32, bias=False)
        self.layer2 = nn.Linear(32, 20, bias=False)
        self.layer3 = nn.Linear(20, 1)
        self.bm1 = nn.BatchNorm1d(32)
        self.bm2 = nn.BatchNorm1d(20)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.bm1(x)

        x = self.layer2(x)
        x = torch.relu(x)
        x = self.bm2(x)

        x = self.layer3(x)
        return (x)


model = BatchModel()

ds = data.TensorDataset(_global_var_data_x, _global_var_target.float())
d_train, d_test = data.random_split(ds, [0.7, 0.3])
train_data = data.DataLoader(d_train, batch_size=16, shuffle=True)
test_data = data.DataLoader(d_test, batch_size=len(d_test), shuffle=False)

epochs = 5
optimizer = optim.Adam(params=model.parameters(), lr=0.01)
loss_func = nn.BCEWithLogitsLoss()

model.train()
for _e in range(epochs):
    for x, y in train_data:
        predict = model(x)
        #print(predict.size(), y.size())
        loss = loss_func(predict.squeeze(), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
model.eval()
with torch.no_grad():
    x, y = next(iter(test_data))
    predict = model(x).squeeze_()
    #print(((torch.sign(predict)+1)/2), y)

    Q = (((torch.sign(predict)+1)/2) == y).sum().item()/len(y)

