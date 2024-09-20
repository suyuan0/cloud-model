from flask import Flask, request, jsonify
import tensorflow as tf
from socks import method
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# 初始化 Flask 应用
app = Flask(__name__)

# 加载训练好的模型
model = tf.keras.models.load_model('model.h5')

# 定义类别名称
class_names = ["cirrus", "cirrostratus", "cirrocumulus", "altocumulus", "altostratus", "cumulus", "cumulonimbus",
               "nimbostratus", "stratocumulus", "stratus", "contrail"]


# 预处理上传的图片
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(400, 400))  # 图片大小调整为 400x400
    image_array = image.img_to_array(img)
    image_array = np.expand_dims(image_array, axis=0)  # 添加批次维度
    image_array /= 255.0  # 像素值归一化
    return image_array


# 创建图片上传接口
@app.route("/predict", methods=["POST"])
def predict():
    # 确保文件存在
    if 'file' not in request.files:
        return jsonify({'error': "No file uploaded"})

    file = request.files['file']

    # 将图片保存到本地
    img_path = 'uploads/' + file.filename
    file.save(img_path)

    # 预处理图片
    img = preprocess_image(img_path)

    # 进行预测
    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions[0])]

    # 返回结果
    return jsonify({'predicted_class': predicted_class})


if __name__ == "__main__":
    app.run(debug=True)
