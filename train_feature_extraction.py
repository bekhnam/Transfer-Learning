from config import config
from config import create_dataloaders
from imutils import paths
from torchvision.models import resnet50
from torchvision import transforms
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

def main():
    # define augmentation pipelines
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

    # create data loaders
    (trainDS, trainLoader) = create_dataloaders.get_dataloader(config.TRAIN,
        transforms=trainTransform,
        batchSize=config.FEATURE_EXTRACTION_BATCH_SIZE)

    (valDS, valLoader) = create_dataloaders.get_dataloader(config.VAL,
        transforms=valTransform,
        shuffle=False,
        batchSize=config.FEATURE_EXTRACTION_BATCH_SIZE)

    # load up the resnet50 model
    model = resnet50(pretrained=True)

    # set parameters to non-trainable
    for param in model.parameters():
        param.requires_grad = False

    # append a new classification top to our feature extractor and pop it on to the current device
    modelOutputFeats = model.fc.in_features
    model.fc = nn.Linear(modelOutputFeats, len(trainDS.classes))
    model = model.to(config.DEVICE)

    # initialize loss function and optimizer
    lossFunc = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.fc.parameters(), lr=config.LR)

    # calculate steps per epoch for training and validation set
    trainSteps = len(trainDS) // config.FEATURE_EXTRACTION_BATCH_SIZE
    valSteps = len(valDS) // config.FEATURE_EXTRACTION_BATCH_SIZE

    # initialize a dictionary to store training history
    H = {"train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []}

    # loop over epochs
    print("[INFO] training the network ...")
    startTime = time.time()
    for e in tqdm(range(config.EPOCHS)):
        model.train()

        totalTrainLoss = 0
        totalValLoss = 0

        trainCorrect = 0
        valCorrect = 0

        # loop over the training set
        for (i, (x, y)) in enumerate(trainLoader):
            (x ,y) = (x.to(config.DEVICE), y.to(config.DEVICE))
            pred = model(x)
            loss = lossFunc(pred, y)

            loss.backward()
            if (i+2) % 2 == 0:
                opt.step()
                opt.zero_grad()
            
            totalTrainLoss += loss
            trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
            torch.cuda.empty_cache()
        
        # switch off autograd
        with torch.no_grad():
            model.eval()

            # loop over the validation set
            for (x, y) in valLoader:
                (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
                pred = model(x)

                totalValLoss += lossFunc(pred, y)
                valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps

        # calculate the training and validation accuracy
        trainCorrect = trainCorrect / len(trainDS)
        valCorrect = valCorrect / len(valDS)

        # update training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["train_acc"].append(trainCorrect)
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["val_acc"].append(valCorrect)

        # print the model training and validation info
        print("[INFO] EPOCH: {}/{}".format(e+1, config.EPOCHS))
        print("Train Loss: {:.6f}, Train Accuracy: {:.4f}".format(avgTrainLoss, trainCorrect))
        print("Val Loss: {:.6f}, Val Accuracy: {:.4f}".format(avgValLoss, valCorrect)) 

    # serialize the model to disk
    torch.save(model, config.WARMUP_MODEL)

    # display the total time needed to perform the training
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.0f}m {:.0f}s".format((endTime - startTime)//60, (endTime - startTime)%60))

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["val_loss"], label="val_loss")
    plt.plot(H["train_acc"], label="train_acc")
    plt.plot(H["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(config.WARMUP_PLOT)

    
if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()