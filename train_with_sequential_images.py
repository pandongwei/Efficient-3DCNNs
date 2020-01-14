import torch
import torch.backends.cudnn as cudnn
from torch import optim
from tensorboardX import SummaryWriter
import numpy as np
from torch.utils.data import Dataset,DataLoader
import os
from models.mobilenetv2 import *
import json
import time
import copy
from datetime import datetime
import cv2

DATA_CLASS_NAMES = {
    "bicycle-lane":0,
    "bicycle-lane-and-pedestrian":1,
    "car-lane":2,
    "pedestrian":3
}

class SequentialDataset(Dataset):
    '''
    generate the sequential image dataset that several images as one input
    '''

    def __init__(self, root_path, images_len=10, frame_count = 10, height = 224, width = 224,rescale = None):
        self.root_path = root_path
        self.images_len = images_len
        self.fnames, self.labels = [], []
        self.frame_count = frame_count
        self.height = height
        self.width = width
        self.rescale = rescale
        part = []
        for label in sorted(os.listdir(root_path)):
            i = 0
            for fname in sorted(os.listdir(os.path.join(root_path, label))):
                if i < 10:
                    part.append(os.path.join(root_path, label, fname))
                    i += 1
                else:
                    self.labels.append(DATA_CLASS_NAMES.get(label))
                    self.fnames.append(part)
                    part = []
                    i = 0
        assert len(self.labels) == len(self.fnames)


    def __getitem__(self, index):
        buffer = np.empty((self.frame_count, self.height, self.width, 3), np.dtype('float32'))
        for i,frame_name in enumerate(self.fnames[index]):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            if i < 10:
                buffer[i] = frame
            else:
                break
        labels = np.array(self.labels[index])
        if self.rescale is not None:
            buffer = buffer*self.rescale
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def __len__(self):
        return len(self.fnames)

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    since = time.time()

    #use tensorboard
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir = log_dir)

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

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

            # deep copy the model,这里只保存了最好的模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            if phase == 'train:':
                writer.add_scalar('train_acc', epoch_acc, epoch)
                writer.add_scalar('train_loss', epoch_loss, epoch)
            else:
                writer.add_scalar('valid_acc', epoch_acc, epoch)
                writer.add_scalar('valid_loss', epoch_loss, epoch)
        torch.save(model.state_dict(), log_dir+'/final_model.pkl')
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    writer.close()
    return model, val_acc_history

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  ## todo

    with open('config/config.json', 'r') as f:
        cfg = json.load(f)
    batch_size = int(cfg['batch'])
    num_epochs = int(cfg['epochs'])
    num_classes = int(cfg['class_number'])
    shape = (int(cfg['height']), int(cfg['width']), 3)
    learning_rate = cfg['learning_rate']
    pre_weights = cfg['weights']
    train_dir = cfg['train_dir']
    eval_dir = cfg['eval_dir']
    test_dir = cfg['test_dir']

    model = get_model(num_classes=num_classes, sample_size=shape[0], width_mult=1.0)
    model = model.cuda()

    #load weights if it has
    if pre_weights and os.path.exists(pre_weights):
        weights = torch.load(pre_weights)
        model.load_state_dict(weights)

    '''
    model_path = r'./logs/fit/1/jester_mobilenetv2_1.0x_RGB_16_best.pth'
    checkpoint = torch.load(model_path)
    #load part of the weight because of finetune
    pretrained_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
    '''

    '''
    train_dataloader = DataLoader(SequentialDataset(root_path=train_dir,images_len=10,rescale=1/255.), batch_size=batch_size,shuffle=True, num_workers=4)
    val_dataloader = DataLoader(SequentialDataset(root_path=eval_dir,images_len=10,rescale=1/255.), batch_size=batch_size, num_workers=4)
    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)
    '''
    image_dir = {'train': train_dir,
           'val': eval_dir}
    dataloaders_dict = {
        x: DataLoader(SequentialDataset(root_path=image_dir[x],images_len=10,rescale=1/255.), batch_size=batch_size, shuffle=True, num_workers=4) for x in
        ['train', 'val']}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    # Send the model to GPU
    model = model.to(device)

    params_to_update = model.parameters()
    print("Params to learn:")

    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

    # Observe that all parameters are being optimized
    #optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9)
    optimizer = optim.Adam(params_to_update,lr=learning_rate)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model, hist = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)
    path = 'models/model_final.pkl'
    torch.save(model.state_dict(), path)
    '''
    #test the result
    running_loss = 0.0
    running_corrects = 0.0
    test_dataloader = DataLoader(SequentialDataset(root_path= test_dir, images_len=10, rescale=1 / 255.),
                                 batch_size=batch_size, num_workers=4)

    test_size = len(test_dataloader.dataset)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]
        running_corrects += torch.sum(preds == labels.data)

    print(test_size)
    print(running_corrects)
    epoch_acc = running_corrects / test_size
    print(epoch_acc)
    '''

if __name__ == "__main__":
    main()
