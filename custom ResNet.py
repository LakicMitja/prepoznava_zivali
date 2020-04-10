from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from PIL.Image import Image
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.nn import functional as F
import PIL
from torch.utils.tensorboard import SummaryWriter
%load_ext tensorboard
%tensorboard --logdir=runs

plt.ion()   # interactive mode

# pregledamo, ali deluje cuda
a = torch.cuda.is_available()
b = torch.backends.cudnn.enabled;
c = torch.version.cuda;

# Normalizacija slik za učenje in validiranje
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'Serengeti Classes Filtered'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}


writerTrain = SummaryWriter(log_dir="./runs/train")
writerVal = SummaryWriter(log_dir="./runs/val")

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(2.001)


# nabor za učenje modela
inputs, classes = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])



def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Vsaka epoha ima fazo za učenje in validiranje
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # nastavljanje modela za učenje
            else:
                model.eval()   # nastavljanje modela za validiranje

            running_loss = 0.0
            running_corrects = 0

            # Iteriranje preko nabora
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # shranjevanje zgodovine, če učimo
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward, če smo v fazi učenja
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistika
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
              writerTrain.add_scalar('Loss', epoch_loss, epoch)
              writerTrain.add_scalar('Accuracy', epoch_acc, epoch)

            if phase == 'val':
              writerVal.add_scalar('Loss', epoch_loss, epoch)
              writerVal.add_scalar('Accuracy', epoch_acc, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # globoko kopiranje modela
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # nalaganje najboljših uteži modela
    model.load_state_dict(best_model_wts)
    return model


# prikaz slik in predviden razred za vsako sliko
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)

            return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # 224x224
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 112x112

        x = self.layer1(x)  # 56x56
        x = self.layer2(x)  # 28x28
        x = self.layer3(x)  # 14x14
        x = self.layer4(x)  # 7x7

        x = self.avgpool(x)  # 1x1
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet16(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 1], **kwargs)
    return model


model_ft = resnet16(num_classes=60)

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Pregledamo, da so vsi parametri optimizirani
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# Klicanje metode za učenje modela
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=45)

visualize_model(model_ft)


# Napovedovanje
idx_to_class = {
    0: "aardvark", 1: "aardwolf",
    2: "baboon", 3: "bat",
    4: "batearedfox", 5: "buffalo",
    6: "bushbuck", 7: "caracal",
    8: "cattle", 9: "cheetah",
    10: "civet", 11: "dikdik",
    12: "duiker", 13: "eland",
    14: "elephant", 15: "fire",
    16: "gazellegrants", 17: "gazellethomsons",
    18: "genet", 19: "giraffe",
    20: "guineafowl", 21: "hare",
    22: "hartebeest", 23: "hippopotamus",
    24: "honeybadger", 25: "human",
    26: "hyenabrown", 27: "hyenaspotted",
    28: "hyenastriped", 29: "impala",
    30: "insectspider", 31: "jackal",
    32: "koribustard", 33: "kudu",
    34: "leopard", 35: "lioncub",
    36: "lionfemale", 37: "lionmale",
    38: "mongoose", 39: "monkeyvervet",
    40: "ostrich", 41: "otherbird",
    42: "pangolin", 43: "porcupine",
    44: "reedbuck", 45: "reptiles",
    46: "rhinoceros", 47: "rodents",
    48: "secretarybird", 49: "serval",
    50: "steenbok", 51: "topi",
    52: "vulture", 53: "warthog",
    54: "waterbuck", 55: "wildcat",
    56: "wilddog", 57: "wildebeest",
    58: "zebra", 59: "zorilla"
}

def predict(model, test_image_name):
    transform = data_transforms['val']

    test_image = PIL.Image.open(test_image_name)
    plt.imshow(test_image)

    test_image_tensor = transform(test_image)

    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)

    with torch.no_grad():
        model.eval()
        # Model prikaže verjetnosti
        out = model(test_image_tensor)
        ps = torch.exp(out)
        topk, topclass = ps.topk(1, dim=1)
        print("Output class :  ", idx_to_class[topclass.cpu().numpy()[0][0]])


# Shranjevanje naučenega modela
PATH = './ResNet_Custom.pth'
torch.save(model_ft.state_dict(), PATH)


# Nalaganje modela
model_ft = resnet16()
num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs, len(class_names))
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft.load_state_dict(torch.load('./ResNet_Custom.pth'))

# Prepoznava živali z loadanim modelom
visualize_model(model_ft)
