import pandas as pd 
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cpu.is_available() else 'cpu'

# 데이터셋 설정
test_set = pd.read_csv("mnist_test.csv")
train_set = pd.read_csv("mnist_train.csv")

# 데이터 클래스 설정
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label = torch.tensor(self.data.iloc[idx, 0], dtype=torch.long) 
        image = torch.tensor(self.data.iloc[idx, 1:].values/255.0, dtype=torch.float32).reshape(0)
        return label, image

class CNN(nn.Module): 
    def __init__(self):
        super(CNN, self).__init__()
        self.keep_prob = 0.5

        self.layer1 = nn.Sequential(
                        nn.Conv2d(1, 32, kernel_size=3),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer2 = nn.Sequential(
                        nn.Conv2d(32, 64, kernel_size=3),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64, 128, kernel_size=3),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(128, 625, bias=True)
        nn.init.xavier_uniform_(self.fc1.weight)  

        self.layer4 = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob))

        self.fc2 = nn.Linear(625, 10, bias=True)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)  
        out = self.layer2(out)  
        out = self.layer3(out)  
        
        out = out.view(out.size(0), -1) 
        
        out = self.layer4(out) 
        out = self.fc2(out)  
        return out  

    
# 모델, 손실 함수, 옵티마이저, 데이터로더 설정
model = CNN().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=0.0001)

test = CustomDataset(test_set)
train = CustomDataset(train_set)

test_dataloader = DataLoader(test, batch_size=100, shuffle=False)
train_dataloader = DataLoader(train, batch_size=100, shuffle=True)

epoch = 10

# 모델 학습 및 평가 
for e in range(epoch):
    # 평가 모드
    model.eval()
    num_correct = 0
    with torch.no_grad():  
        for batch in tqdm(test_dataloader):
            data_y, data_x = batch

            data_x = data_x.to(device)
            data_y = data_y.to(device)

            output = model(data_x)
            num_correct += (output.argmax(dim=1) == data_y).sum().item()

    print("Accuracy: {:.4f}".format(num_correct / len(test)))

    # 학습 모드
    model.train()
    for batch in tqdm(train_dataloader):
        data_y, data_x = batch

        data_x = data_x.to(device)
        data_y = data_y.to(device)

        optimizer.zero_grad()

        output = model(data_x)

        loss = loss_function(output, data_y)

        loss.backward()

        optimizer.step()

    print("Epoch: {}, Loss: {:.4f}".format(e, loss.item()))
