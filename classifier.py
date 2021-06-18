import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from torch.utils.data import DataLoader
from config import DATASET_PARAMETERS, NETWORKS_PARAMETERS
from parse_dataset import get_dataset
from network import get_network
from utils import Meter, cycle, save_model


# another model---- pretrained resnet model
model_res = models.resnet18(pretrained = True)


num = 0
for param in model_res.parameters():
  if num<40:
    param.requires_grad = False
  num +=1
print(num) # 61

num_features = model_res.fc.in_features
   

# model_res.fc = nn.Linear(num_features,1251)

model_res.fc = nn.Sequential(nn.Linear(num_features,1024),
                             nn.ReLU(),
                             nn.Linear(1024,5120),
                             nn.ReLU(),
                             nn.Linear(5120,2048),
                             nn.ReLU(),
                             nn.Linear(2048,1251)
                             )



# dataset and dataloader
print('Parsing your dataset...')
voice_list, face_list, id_class_num = get_dataset(DATASET_PARAMETERS)
NETWORKS_PARAMETERS['c']['output_channel'] = id_class_num

print('Preparing the datasets...')
voice_dataset = DATASET_PARAMETERS['voice_dataset'](voice_list,
                               DATASET_PARAMETERS['nframe_range'])
face_dataset = DATASET_PARAMETERS['face_dataset'](face_list)

print('Preparing the dataloaders...')
collate_fn = DATASET_PARAMETERS['collate_fn'](DATASET_PARAMETERS['nframe_range'])
voice_loader = DataLoader(voice_dataset, shuffle=True, drop_last=True,
                          batch_size=DATASET_PARAMETERS['batch_size'],
                          num_workers=DATASET_PARAMETERS['workers_num'],
                          collate_fn=collate_fn)
face_loader = DataLoader(face_dataset, shuffle=True, drop_last=True,
                         batch_size=DATASET_PARAMETERS['batch_size'],
                         num_workers=DATASET_PARAMETERS['workers_num'])


voice_iterator = iter(cycle(voice_loader))
face_iterator = iter(cycle(face_loader))

start = time.time()
# best_accuracy = 0.0
epochs = 100

def calc_accuracy(prediction, label):
    with torch.no_grad():
        _, pred2label = prediction.max(dim=1)
        # print(pred2label, label)
        same = (pred2label == label).float()
        accuracy = same.sum().item() / same.numel()
    return accuracy


model_res.to('cuda')
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model_res.parameters(), lr=0.001,weight_decay = 1e-5)
for epoch in range(epochs):
    print(epoch)
    train_accuracy_list, valid_accuracy_list = [], []
    train_total_loss, valid_total_loss = 0.0, 0.0
    model_res.train()
    # load mini-batches and do training
    face, face_label = next(face_iterator)
    batch = face.cuda()
    label = face_label.cuda()

    prediction = model_res(batch)
    loss = criterion(prediction, label)
    print(loss)
    optim.zero_grad()
    loss.backward()
    optim.step()
    train_accuracy_list.append(calc_accuracy(prediction, label))
    train_total_loss += loss.item()
    
    train_total_accuracy = sum(train_accuracy_list) / len(train_accuracy_list)
    # for validation, you do not need to calculate the gradient
    with torch.no_grad():
        model_res.eval()
        # load mini-batches and do validation
        face, face_label = next(face_iterator)
        batch = face.cuda()
        label = face_label.cuda()

        prediction = model_res(batch)
        loss = criterion(prediction, label)
        
        valid_accuracy_list.append(calc_accuracy(prediction, label))
        valid_total_loss += loss.item()
        
        valid_total_accuracy = sum(valid_accuracy_list) / len(valid_accuracy_list)

    print(f"""{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} || [{epoch}/{epochs}], train_loss = {train_total_loss:.4f}, train_accuracy = {train_total_accuracy:.2f}, valid_loss = {valid_total_loss:.4f}, valid_accuracy = {valid_total_accuracy:.2f}""")

elapsed = time.time() - start
print(f"End of training, elapsed time : {elapsed // 60} min {elapsed % 60} sec.")
save_model(model_res, '/models/Our_classifier.pth')
