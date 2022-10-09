from os import lseek
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn  # pylint: disable = all
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms
from io import BytesIO

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"

app = FastAPI()
#TODO 命令行输入device
device = 'cpu'

class Net(nn.Module):
    """ Network architecture. """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):  # pylint: disable = arguments-differ
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def load_model(modelpath:str, device:str):
    model = Net()
    model.load_state_dict(torch.load(modelpath))
    model.to(device)
    model.eval()
    return model

#TODO 修改文件名
modelPath = '/mnt/admin/pytorch/model/model_cpu.dat'
model = load_model(modelPath,device)





transform_vaild = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28,28)),
    transforms.ToTensor()
])
# image = './MNIST/0.jpg'
# img = Image.open(image)
# img_ = transform_vaild(img).unsqueeze(0)
# img_ = img_.to(device)
# outputs = model(img_)
# per = F.softmax(outputs,dim=1)


def serve(model, pic:Image.Image):
    inputs =transform_vaild(pic).unsqueeze(0)
    inputs = inputs.to(device)
    outputs = model(inputs)
    outputs = F.softmax(outputs,dim=1)
    max = outputs[0][0]
    maxnum = 0
    for i, j in enumerate(outputs[0]):
        if j > max:
            max = j
            maxnum = i
    print(outputs[0])
    return maxnum


@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

@app.post("/mnist")
async def serve_api(file: UploadFile = File(...)):
    print('1')
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    pics = read_imagefile(await file.read())
    # picspath = './MNIST/0.jpg'
    # pics = Image.open(picspath)
    res = serve(model, pics)
    return {"result": res}


# uvicorn main:app --host '0.0.0.0' --port 8123 --reload

if __name__ == "__main__":
    # model_path = './models/model_gpu.dat'
    uvicorn.run(app=app, host='0.0.0.0', port=8124)