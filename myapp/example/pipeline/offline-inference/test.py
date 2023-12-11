import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def predict(image_path, model):
    # 加载图像并在 GPU 上进行推理
    image_tensor = load_image(image_path).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
    return preds.item()


if __name__ == '__main__':
    # 检查 GPU 可用性
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 加载预训练的 ResNet-18 模型
    # model = models.resnet50(pretrained=True).to(device)
    model = models.resnet50(pretrained=False)
    model.load_state_dict(torch.load('resnet50.pth'))
    model = model.to(device)
    # 切换到评估模式
    model.eval()
    # 指定图像路径
    image_path = "test.jpg"

    # 对图像进行推理
    for image_path in open('images_url.txt').readlines():
        image_path = image_path.replace('\n','')
        prediction = predict(image_path, model)
        print("Predicted class:", prediction)