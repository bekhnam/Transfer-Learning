from config import config, create_dataloaders
from imutils import paths
from torchvision.models import resnet50
from torchvision import transforms
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import shutil
import torch
import time
import os

def main():
    trainTransform = transforms.Compose([
        transforms.RandomResizedCrop(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    valTransform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])

    (trainDS, trainLoader) = create_dataloaders.get_dataloader(config.TRAIN,
        transforms=trainTransform, batchSize=config.FINETUNE_BATCH_SIZE)
    (valDS, valLoader) = create_dataloaders.get_dataloader(config.VAL,
        transforms=valTransform, batchSize=config.FINETUNE_BATCH_SIZE, shuffle=False)

    model = resnet50(pretrained=True)
    numFeatures = model.fc.in_features

    # set parameters of batch normalization modules as non-trainable
    for module, param in zip(model.modules(), model.parameters()):
        if isinstance(module, nn.BatchNorm2d):
            param.requires_grad = False

    # define the network head
    headModel = nn.Sequential(
        nn.Linear(numFeatures, 512),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, len(trainDS.classes))
    )
    model.fc = headModel

    model = model.to(config.DEVICE)

    lossFunc = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=config.LR)

    trainSteps = len(trainDS) // config.FINETUNE_BATCH_SIZE
    valSteps = len(valDS) // config.FINETUNE_BATCH_SIZE

    H = {"train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []}

    # loop over epochs
    print("[INFO] training the network...")
    startTime = time.time()
    for e in tqdm(range(config.EPOCHS)):
        model.train()

        totalTrainLoss = 0
        totalValLoss = 0

        trainCorrect = 0
        valCorrect = 0

        for (i, (x, y)) in enumerate(trainLoader):
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
            pred = model(x)
            loss = lossFunc(pred, y)

            loss.backward()
            if (i+2) % 2 == 0:
                opt.step()
                opt.zero_grad()
            
            totalTrainLoss += loss
            trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
            torch.cuda.empty_cache()

        with torch.no_grad():
            model.eval()
            for (x, y) in valLoader:
                x, y = x.to(config.DEVICE), y.to(config.DEVICE)
                pred = model(x)
                totalValLoss += lossFunc(pred, y)
                valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        trainCorrect = trainCorrect / len(trainDS)
        valCorrect = valCorrect / len(valDS)

        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["train_acc"].append(trainCorrect)
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["val_acc"].append(valCorrect)

        print("[INFO] EPOCH: {}/{}".format(e+1, config.EPOCHS))
        print("Train Loss: {:.6f}, Train Accuracy: {:.4f}".format(avgTrainLoss, trainCorrect))
        print("Val Loss: {:.6f}, Val Accuracy: {:.4f}".format(avgValLoss, valCorrect))

    elapsed = time.time() - startTime
    print("[INFO] total time taken to train the model: {:.0f}m {:.0f}s".format(elapsed//60, elapsed%60))

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["train_acc"], label="train_acc")
    plt.plot(H["val_loss"], label="val_loss")
    plt.plot(H["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(config.FINETUNE_PLOT)

    torch.save(model, config.FINETUNE_MODEL)

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()