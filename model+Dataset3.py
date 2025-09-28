import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import sklearn
from sklearn.datasets import load_diabetes

# Загрузка датасета
diabetes = load_diabetes()

# Форма данных и целевой переменной
print(diabetes.data.shape)  # (442, 10)
print(diabetes.target.shape)  # (442,)

# Создание глобальных пееременных для тестирования
_global_var_data_x = torch.tensor(diabetes.data, dtype=torch.float32)
_global_var_target = torch.tensor(diabetes.target, dtype=torch.float32)


class dadaset(data.Dataset):
    def __init__(self):
        self.data = _global_var_data_x
        self.target = _global_var_target
        self.length = len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.target[item]

    def __len__(self):
        return self.length


class DiabetModel(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.tanh(x)
        x = self.layer2(x)
        return x


epoch = 10
batch_size = 8

model = DiabetModel(10,64,1)
model.train()

d_train = dadaset()
#print(d_train[0:3])
train_data = data.DataLoader(d_train, batch_size, shuffle=True)

optimizer = optim.RMSprop(params=model.parameters(), lr=0.01)
loss_func = nn.MSELoss()

for _ in range(epoch):
    for x, y in train_data:
        #print(x.shape, y.shape)
        predict = model(x)
        loss = loss_func(predict.squeeze(), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()
Q = loss_func(model(d_train.data).squeeze(), d_train.target).item()
#print(model(d_train.data).size(), d_train.target.size())
print(Q)
Q1 = ((model(d_train.data).squeeze()-d_train.target)**2).sum().item()/len(d_train.data)
print(Q1)