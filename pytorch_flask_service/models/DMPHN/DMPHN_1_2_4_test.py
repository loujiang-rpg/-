import torch
from torch.autograd import Variable
import math
import models.DMPHN.model as model
import torchvision
from torchvision import transforms
from PIL import Image


def save_images(images):
    filename = './out.png'
    torchvision.utils.save_image(images, filename)


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, 0.5 * math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def main(img):
    print("init data folders")

    encoder_lv1 = model.Encoder().apply(weight_init).cuda()
    encoder_lv2 = model.Encoder().apply(weight_init).cuda()
    encoder_lv3 = model.Encoder().apply(weight_init).cuda()

    decoder_lv1 = model.Decoder().apply(weight_init).cuda()
    decoder_lv2 = model.Decoder().apply(weight_init).cuda()
    decoder_lv3 = model.Decoder().apply(weight_init).cuda()

    encoder_lv1.load_state_dict(torch.load('models/DMPHN/encoder_lv1.pkl'))
    encoder_lv2.load_state_dict(torch.load('models/DMPHN/encoder_lv2.pkl'))
    encoder_lv3.load_state_dict(torch.load('models/DMPHN/encoder_lv3.pkl'))
    decoder_lv1.load_state_dict(torch.load('models/DMPHN/decoder_lv1.pkl'))
    decoder_lv2.load_state_dict(torch.load('models/DMPHN/decoder_lv2.pkl'))
    decoder_lv3.load_state_dict(torch.load('models/DMPHN/decoder_lv3.pkl'))

    with torch.no_grad():
        images_lv1 = transforms.ToTensor()(img)
        images_lv1 = Variable(images_lv1 - 0.5).unsqueeze(0).cuda()
        H = images_lv1.size(2)
        W = images_lv1.size(3)

        images_lv2_1 = images_lv1[:, :, 0:int(H / 2), :]
        images_lv2_2 = images_lv1[:, :, int(H / 2):H, :]
        images_lv3_1 = images_lv2_1[:, :, :, 0:int(W / 2)]
        images_lv3_2 = images_lv2_1[:, :, :, int(W / 2):W]
        images_lv3_3 = images_lv2_2[:, :, :, 0:int(W / 2)]
        images_lv3_4 = images_lv2_2[:, :, :, int(W / 2):W]

        feature_lv3_1 = encoder_lv3(images_lv3_1)
        feature_lv3_2 = encoder_lv3(images_lv3_2)
        feature_lv3_3 = encoder_lv3(images_lv3_3)
        feature_lv3_4 = encoder_lv3(images_lv3_4)
        feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3)
        feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3)
        feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
        residual_lv3_top = decoder_lv3(feature_lv3_top)
        residual_lv3_bot = decoder_lv3(feature_lv3_bot)

        feature_lv2_1 = encoder_lv2(images_lv2_1 + residual_lv3_top)
        feature_lv2_2 = encoder_lv2(images_lv2_2 + residual_lv3_bot)
        feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2) + feature_lv3
        residual_lv2 = decoder_lv2(feature_lv2)

        feature_lv1 = encoder_lv1(images_lv1 + residual_lv2) + feature_lv2
        deblur_image = decoder_lv1(feature_lv1)

        # save_images(deblur_image.data + 0.5)
        return deblur_image.data + 0.5
