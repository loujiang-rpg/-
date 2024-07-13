import torch
import torchvision
from .PINet_model import PINet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def predict(image):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(size=224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))
    ])
    img = transforms(image).float().unsqueeze(0)

    model = PINet().to(device)  # 模型
    # load model weights
    state_dict_load = torch.load('./models/PINet/PINet.pt')  # 加载训练参数模型
    model.load_state_dict(state_dict_load)  # 给模型加载参数
    model.eval()

    out = model(torch.tensor(img.cuda())).item()
    return out
