from config import config, create_dataloaders
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn
import argparse
import torch

def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="path to the trained model")
    args = vars(ap.parse_args())

    testTransform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])

    # calculate the inverse mean and standard deviation
    invMean = [-m/s for (m, s) in zip(config.MEAN, config.STD)]
    invStd = [1/s for s in config.STD]

    # de-normalization transform
    deNormalize = transforms.Normalize(mean=invMean, std=invStd)

    # initialize test dataset and data loader
    print("[INFO] loading the dataset...")
    (testDS, testLoader) = create_dataloaders.get_dataloader(config.VAL,
        transforms=testTransform, batchSize=config.PRED_BATCH_SIZE, shuffle=True)

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = "cpu"

    # load the model
    print("[INFO] loading the model...")
    model = torch.load(args["model"], map_location=map_location)

    model.to(config.DEVICE)
    model.eval()

    batch = next(iter(testLoader))
    (images, labels) = (batch[0], batch[1])

    fig = plt.figure("Results", figsize=(10, 10))

    with torch.no_grad():
        images = images.to(config.DEVICE)
        preds = model(images)
        for i in range(0, config.PRED_BATCH_SIZE):
            ax = plt.subplot(config.PRED_BATCH_SIZE//2, config.PRED_BATCH_SIZE//2 , i+1)
            image = images[i]
            image = image.to("cpu")
            image = deNormalize(image).cpu().numpy()
            image = (image*255).astype("uint8")
            image = image.transpose((1, 2, 0))

            idx = labels[i].cpu().numpy()
            gtLabel = testDS.classes[idx]

            pred = preds[i].argmax().cpu().numpy()
            predLabel = testDS.classes[pred]

            info = "Ground Truth: {}, Predicted: {}".format(gtLabel, predLabel)
            plt.imshow(image)
            plt.title(info)
            plt.axis("off")
        # show the plot
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
