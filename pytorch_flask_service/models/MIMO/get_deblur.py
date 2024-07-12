import torch
from PIL import Image

from .MIMOUNet import build_net
from torchvision.transforms import functional as F


def get(img):
    # model = build_net('MIMO-UNet')
    # state_dict = torch.load('./models/MIMO/MIMO-UNet.pkl')
    # model.load_state_dict(state_dict['model'])
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    # model.eval()
    #
    # img = F.to_tensor(img).unsqueeze(0)
    # input_img = img.to(device)
    # _ = model(input_img)
    # pred = model(input_img)[2]
    #
    # pred_clip = torch.clamp(pred, 0, 1)
    #
    # pred_clip += 0.5 / 255
    # pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
    # return pred  # 本地跑不动
    return img
