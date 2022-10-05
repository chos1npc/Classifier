import torch
from torch import nn
import torch.optim as optim
import torchvision.transforms as T
from torch.nn import functional as F
import torchvision.models
from torch.utils.data import DataLoader
from dataset import AICUPdataset
import os
import matplotlib.pyplot as plt
import time 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#datapath
train_file = "train.csv"
valid_file = "valid.csv"
image_dir = "/media/mmlab206/YT8M-4TB/aicup_data/aicup"


n_epoches = 100
image_size = 160
batch = 64
# Configure data loader
train_dataset = AICUPdataset(
    train_file,
    image_dir,
    transform=T.Compose([
        T.Resize([image_size,image_size]),
        # T.RandomPerspective(distortion_scale=0.6,p=0.5),
        T.RandomRotation(degrees=(0,100)),
        T.RandomHorizontalFlip(0.5),
        T.ToTensor(),
    ]),
)
valid_dataset = AICUPdataset(
    valid_file,
    image_dir,
    transform= T.Compose([
        T.Resize([image_size, image_size]),
        T.ToTensor(),
    ]),

)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch,
    shuffle=True,
)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=8,
)



model = torchvision.models.resnet101(num_classes=33)
# numFit = model.fc.in_features
# model.fc = nn.Linear(numFit, 33)

# use all the available gpus
model = nn.DataParallel(model)
model.to(device)

criterion = nn.CrossEntropyLoss()


# Optimizer
Optimizer = optim.Adam(model.parameters())

# Training
train_size = len(train_dataset)
valid_size = len(valid_dataset)
loss_temp = 0
print(f"Training dataset: {train_size} images")
for epoch in range(n_epoches):
    train_loss = 0
    valid_loss = 0
    train_correct = 0
    valid_correct = 0
    start_time =time.time()

    model.train()
    for i, (images, labels) in enumerate(train_dataloader):

        images = images.to(device)
        label = labels.to(device)
        # print(images)
        # print(labels)
        
        out = model(images)
        # demo result
        # print(out)
        # print(out.shape)
        # exit()

        loss = criterion(out, label)
        # demo loss
        # print(loss)
        # exit()

        train_loss += loss.item()
        _, gt = torch.max(label, 1)
        _, pred = torch.max(out, 1)


        num_correct = (pred==gt).sum()
        train_correct += num_correct

        # Backward
        Optimizer.zero_grad()
        loss.backward()
        Optimizer.step()

        if i % 10 == 0:
            used_time = time.time()- start_time
            start_time = time.time()
            print(f"[{epoch:3}/{n_epoches}] {i:3}/{train_size//batch} loss: {loss.item():.2f}  use {used_time:.2f} second")
    # Evaluation
    model.eval()
    print(f"Start validation.")
    start_time = time.time()

    for i, (images, labels) in enumerate(valid_dataloader):

        images = images.to(device)
        label = labels.to(device)

        out = model(images)
        loss = criterion(out, label)
        valid_loss += loss.item()
        _, gt = torch.max(label, 1)
        _, pred = torch.max(out, 1)

        num_correct = (pred==gt).sum()
        valid_correct += num_correct

    train_acc = train_correct / train_size
    valid_acc = valid_correct / valid_size
    used_time = time.time() - start_time
    print(f"[{epoch:3}/{n_epoches}] train loss: {train_loss:.2f}; valid loss: {valid_loss:.2f}; train acc:{train_acc*100:.2f}%; valid acc: {valid_acc*100:.2f}% ; use {used_time:.2f} second")

    # record
    with open("log.txt", "a") as f:
        if epoch == 0:
            f.write(f"Epoch{n_epoches}, train_loss, valid_loss, train_acc, valid_acc\n")
            f.write(f"{epoch}, {train_loss:.2f}, {valid_loss:.2f}, {train_acc*100:.2f}, {valid_acc*100:.2f}\n")
        else:
            f.write(f"{epoch}, {train_loss:.2f}, {valid_loss:.2f}, {train_acc*100:.2f}, {valid_acc*100:.2f}\n")
    
    
    # Save model
    torch.save(model.module.state_dict(), "latest_model.pth")
    if epoch == 0:
        loss_temp = valid_loss
    else:
        if valid_loss < loss_temp:
            torch.save(model.module.state_dict(), "best_model.pth")
            loss_temp = valid_loss
