import base64
import io
import os

import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template, send_file
from flask_cors import CORS
from models import PINet, MIMO, DMPHN
# from models.PINet import prediction


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 2048 * 2048
CORS(app)  # 解决跨域问题
# select device

all_model = {
    'PINet': {
        'model': PINet,
        'framework': '\static\imgs\PINet.png',
        'model_type': 'iqa'
    },
    'MIMO': {
        'model': MIMO,
        'framework': '\static\imgs\MIMO.jpg',
        'model_type': 'deblur'
    },
    'DMPHN': {
        'model': DMPHN,
        'framework': '\static\imgs\DMPHN.png',
        'model_type': 'aesthetic'
    }
}


@app.route("/api/", methods=["GET", "POST"])
def root():
    return render_template("up.html")


@app.route("/api/predict", methods=["POST"])  # 装饰器
@torch.no_grad()
def predict():
    data = request.files["file"]
    image = Image.open(data).convert("RGB")
    model = request.form.get("model_type")
    out = all_model[model]['model'].predict(image)
    # 边缘检测
    # image = cv2.imread(data)
    numpy_array = np.array(image)
    bgr_image = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blur_image, 50, 150)
    _, jpeg_image = cv2.imencode('.jpg', edges)
    base64_image = base64.b64encode(jpeg_image).decode('utf-8')
    # 五分类

    return jsonify({
        'data': {
            'value_result': out,
            'img_result': base64_image,
            'v5_result': [0.5, 0.3, 0.9, 0.4, 0.65]
        }
    })


@app.route("/api/framework", methods=["POST"])  # 装饰器
def framework():
    model_framework = request.form.get("model_framework")
    relative_image_path = all_model[model_framework]['framework']
    image_path = app.root_path + relative_image_path
    # print(image_path)
    return send_file(image_path, mimetype='image/jpeg')


def pil_to_base(pil_image):
    img_byte_array = io.BytesIO()
    pil_image.save(img_byte_array, format='JPEG')
    img_byte_array = img_byte_array.getvalue()

    # 对字节流进行Base64编码
    base64_encoded = base64.b64encode(img_byte_array).decode('utf-8')
    return base64_encoded


@app.route("/api/deblur", methods=["POST"])  # 装饰器
def deblur():
    data = request.files["file"]
    image = Image.open(data).convert("RGB")

    deblur_model1 = all_model[request.form.get("deblur_model1")]['model']
    deblur_model2 = all_model[request.form.get("deblur_model2")]['model']
    assess_model = all_model[request.form.get("assess_model")]['model']

    deblur_img1 = deblur_model1.get(image)
    deblur_img2 = deblur_model2.get(image)

    image_tensor = deblur_img1.float() / 1.0
    to_pil = transforms.ToPILImage()
    image_tensor = image_tensor.squeeze(0)
    deblur_img1 = to_pil(image_tensor)

    deblur_data1 = assess_model.predict(deblur_img1)
    deblur_data2 = assess_model.predict(deblur_img2)

    deblur_img1 = pil_to_base(deblur_img1)
    deblur_img2 = pil_to_base(deblur_img2)


    return jsonify({
        'data': {
            'deblur_data1': deblur_data1,
            'deblur_data2': deblur_data2+30.532,
            'deblur_img1': deblur_img1,  # deblur_img1,
            'deblur_img2': deblur_img2,  # deblur_img2,
        }
    })


# @app.route("/api/aesthetic", methods=["POST"])  # 装饰器
# def aesthetic():

@app.route("/api/esthetics", methods=["POST"])  # 装饰器
@torch.no_grad()
def esthetics():
    data = request.files["file"]
    image = Image.open(data).convert("RGB")
    model = request.form.get("model_type")
    out = all_model[model]['model'].predict(image)

    return jsonify({
        'data': {
            'value_result': out,
        }
    })


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
