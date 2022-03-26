import base64,re
from PIL import Image
from io import BytesIO
import torch,os
from torchvision import transforms
import torch.utils.data as data
import torch.nn as nn

class net(nn.Module):#11
    def __init__(self):
        super(net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, 5), 
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*18*18, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1,16*18*18)
        x = self.fc(x)
        return x

TRANSFORM = transforms.Compose([transforms.Resize((84,84)),
                                transforms.ToTensor()
                               ])

def base64_to_image(base64_str):
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    return img

def check_ba64(base_64_pic):
    img=base64_to_image(base_64_pic)
    tr=TRANSFORM(img.convert("RGB"))
    model=net()
    model.load_state_dict(torch.load('./output/[25]-L(0.548469)E(19.167).pt'))
    result=model(tr)
    print(float(result))