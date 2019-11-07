from models.WSCNet import ClassWisePool, ResNetWSL
from torchvision import transforms, models
import torch.nn as nn
import torch
import scripts.configuration as conf

Model_Path = ''

model_ft = models.resnet101(pretrained=True)
pooling = nn.Sequential()
pooling.add_module('class_wise', ClassWisePool(conf.num_feature))
pooling2 = nn.Sequential()
pooling2.add_module('class_wise', ClassWisePool(conf.num_class))
model_ft = ResNetWSL(model_ft, conf.num_class, conf.num_feature, pooling, pooling2)
model_ft = model_ft.to(conf.device)
model_ft.load_state_dict(torch.load(Model_Path))

data_transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(conf.input_size),
    transforms.CenterCrop(conf.input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])





