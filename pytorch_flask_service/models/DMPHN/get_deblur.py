from .DMPHN_1_2_4_test import main
import torchvision


def save_images(images):
    filename = './out.png'
    torchvision.utils.save_image(images, filename)


def get(img):
    out = main(img)
    save_images(out)
    return out
