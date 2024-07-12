from PIL import Image

from DMPHN_1_2_4_test import main

img = Image.open('1.png').convert('RGB')
main(img)
print('jieshu')