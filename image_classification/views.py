import base64
import io
import json
import os
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
import numpy as np
import cv2
from torchvision import models
from torchvision import transforms
from PIL import Image
from django.shortcuts import render
from django.conf import settings

from .forms import ImageUploadForm


# PyTorch-related code from: https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html
# load pretrained DenseNet and go straight to evaluation mode for inference
# load as global variable here, to avoid expensive reloads with each request

# model = CnnModel2()


# data_dir = '/content/drive/MyDrive/data/'
# print('Folders :', os.listdir(data_dir))
# classes = os.listdir(data_dir + "/train")
# print(len(classes),'classes :', classes)


# dataset = ImageFolder(data_dir + '/train', transform=ToTensor())
# print('Size of training dataset :', len(dataset))

# classes = ['badger', 'bat', 'bear', 'bee',
#  'beetle', 'bison', 'boar', 'butterfly',
#   'cat', 'caterpillar', 'cheetah', 'crab',
#    'crow', 'deer', 'dog', 'dolphin', 'duck',
#     'eagle', 'elephant', 'flamingo', 'fox',
#      'goat', 'goldfish', 'goose', 'hamster',
#       'hare', 'hedgehog', 'hippopotamus', 'horse',
#        'hummingbird', 'hyena', 'jellyfish', 'kangaroo',
#         'koala', 'ladybugs', 'leopard', 'lion', 'lizard',
#          'mouse', 'otter', 'owl', 'panda', 'parrot', 'pelecaniformes',
#           'penguin', 'pig', 'pigeon', 'rhinoceros', 'seahorse', 'seal',
#            'shark', 'sheep', 'snake', 'sparrow', 'squid', 'squirrel', 'starfish',
#             'swan', 'turtle', 'whale', 'wolf', 'woodpecker', 'zebra']

classes = ['cats','dogs']


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))



class Block(nn.Module):
    
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x

class ResNet_18(ImageClassificationBase):
    
    def __init__(self, image_channels, num_classes):
        
        super(ResNet_18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def __make_layer(self, in_channels, out_channels, stride):
        
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
            
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride), 
            Block(out_channels, out_channels)
        )
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x 
    
    def identity_downsample(self, in_channels, out_channels):
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(out_channels)
        )


class CnnModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet50(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 2)
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))


model = CnnModel() 
# model = ResNet_18(3,63)
model.load_state_dict(torch.load('./modelcatdog.pth',map_location=torch.device('cpu')))
# model = models.densenet121(pretrained=True)
model.eval()
model.to('cpu')
# load mapping of ImageNet index to human-readable label
# run "python manage.py collectstatic" first!
# json_path = os.path.join(settings.STATICFILES_DIRS[0], "imagenet_class_index.json")

def transform_image(image_bytes):
    """
    Transform image into required DenseNet format: 224x224 with 3 RGB channels and normalized.
    Return the corresponding tensor.
    """
    my_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.ToTensor(),
                                        ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    """For given image bytes, predict the label using the pretrained DenseNet"""
    tensor = transform_image(image_bytes)
    outputs = model.forward(tensor)
    # percent, pred5 = torch.topk(prob,3)

    # _, y_hat = outputs.max(1)
    # predicted_idx = str(y_hat.item())
    # class_name, human_label = classes[predicted_idx]
    # return human_label
    prob =  torch.nn.functional.softmax(outputs,dim=1)
    a,b = torch.max(outputs,1)
    # print(torch.topk(prob,k=3))
    # percent, pred5 = torch.topk(prob,3)
    # print(percent*100)
    # test = pred5.numpy()
    # top3_pred = []
    # for t in test:
    #     print(t)
    # top3_pred.append(t)
    c = b[0].item()
    print(classes[c])
    print(prob)
    d = torch.max(prob,1)
    percent = f'{round(d[0].item()*100,2)}%'
    # print(classes[top3_pred[0][0]])
    # print(classes[top3_pred[0][1]])
    # print(classes[top3_pred[0][2]])
    return classes[c], percent

def index(request):
    image_uri = None
    predicted_label = None

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # passing the image as base64 string to avoid storing it to DB or filesystem
            image = form.cleaned_data['image']
            image_bytes = image.file.read()
            encoded_img = base64.b64encode(image_bytes).decode('ascii')
            image_uri = 'data:%s;base64,%s' % ('image/jpeg', encoded_img)

            # get predicted label
            try:
                predicted_label = get_prediction(image_bytes)
            except RuntimeError as re:
                print(re)
                # predicted_label = "Prediction Error"

    else:
        form = ImageUploadForm()

    context = {
        'form': form,
        'image_uri': image_uri,
        'predicted_label': predicted_label,
    }
    return render(request, 'image_classification/template/index.html', context)