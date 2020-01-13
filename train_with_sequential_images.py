import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import Dataset,DataLoader
import os
from models.mobilenetv2 import *
import json
import time
import copy

DATA_CLASS_NAMES = {
    0: "bicycle-lane",
    1: "bicycle-lane-and-pedestrian",
    2: "car-lane",
    3: "pedestrian"
}

class SequentialDataset(Dataset):
    '''
    generate the sequential image dataset that several images as one input
    '''

    def __init__(self, root_path, images_len=10):
        self.root_path = root_path
        self.images_len = images_len
        self.fnames, self.labels = [], []
        part = []
        for label in sorted(os.listdir(root_path)):
            i = 0
            for fname in sorted(os.listdir(os.path.join(root_path, label))):
                if i < 10:
                    part.append(os.path.join(root_path, label, fname))
                    i += 1
                else:
                    self.labels.append(DATA_CLASS_NAMES.get(label, 0))
                    self.fnames.append(part)
                    part = []
                    i = 0
        assert len(self.labels) == len(self.fnames)


    def __getitem__(self, index):
        buffer = self.fnames[index]
        labels = np.array(self.labels[index])
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def __len__(self):
        return len(self.fnames)

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"  ## todo

    with open('config/config.json', 'r') as f:
        cfg = json.load(f)
    batch = int(cfg['batch'])
    epochs = int(cfg['epochs'])
    num_classes = int(cfg['class_number'])
    shape = (int(cfg['height']), int(cfg['width']), 3)
    learning_rate = cfg['learning_rate']
    pre_weights = cfg['weights']

    model = get_model(num_classes=num_classes, sample_size=shape[0], width_mult=1.)
    model = model.cuda()
    model_path = r'./logs/fit/1/jester_mobilenetv2_1.0x_RGB_16_best.pth'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()


    root_path = "/home/pan/master-thesis-in-mrt/data-sequential/dataset-train"
    train_dataloader = DataLoader(SequentialDataset(root_path=root_path,images_len=10), batch_size=4,shuffle=True, num_workers=4)
    val_dataloader = DataLoader(SequentialDataset(root_path=root_path,images_len=10), batch_size=4, num_workers=4)
    test_dataloader = DataLoader(SequentialDataset(root_path=root_path,images_len=10), batch_size=4, num_workers=4)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5
    )

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    cudnn.benchmark = True



if __name__ == "__main__":
    main()
