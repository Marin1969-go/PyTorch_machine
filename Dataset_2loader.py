import torch
import torch.nn as nn
import torch.utils.data as data


class LinnerudDataset(data.Dataset):
    def __init__(self):
        self.data = torch.tensor(data_x)
        self.targets = torch.tensor(target)
        self.categories = ['Weight', 'Waist', 'Pulse']
        self.features = ['Chins', 'Situps', 'Jumps']
        self.lenght = len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

    def __len__(self):
        return self.lenght


data_x = [[5.0, 162.0, 60.0], [2.0, 110.0, 60.0], [12.0, 101.0, 101.0], [12.0, 105.0, 37.0], [13.0, 155.0, 58.0],
          [4.0, 101.0, 42.0], [8.0, 101.0, 38.0], [6.0, 125.0, 40.0], [15.0, 200.0, 40.0], [17.0, 251.0, 250.0],
          [17.0, 120.0, 38.0], [13.0, 210.0, 115.0], [14.0, 215.0, 105.0], [1.0, 50.0, 50.0], [6.0, 70.0, 31.0],
          [12.0, 210.0, 120.0], [4.0, 60.0, 25.0], [11.0, 230.0, 80.0], [15.0, 225.0, 73.0], [2.0, 110.0, 43.0]]
target = [[191., 36., 50.], [189., 37., 52.], [193., 38., 58.], [162., 35., 62.], [189., 35., 46.], [182., 36., 56.],
          [211., 38., 56.], [167., 34., 60.], [176., 31., 74.], [154., 33., 56.], [169., 34., 50.], [166., 33., 52.],
          [154., 34., 64.], [247., 46., 50.], [193., 36., 46.], [202., 37., 62.], [176., 37., 54.], [157., 32., 52.],
          [156., 33., 54.], [138., 33., 68.]]

d_train = LinnerudDataset()
train_data = data.DataLoader(d_train, batch_size=8, shuffle=False, drop_last=False)

t = iter(train_data)
x, y = next(t)
print(x,y)
x, y = next(t)
print(x,y)