import os
import torch
from flask import Flask, request, jsonify
from torchvision import transforms
from PIL import Image
from torchvision import models

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2()
model.classifier[1] = torch.nn.Linear(model.last_channel, 11)
model.load_state_dict(torch.load("./models/cloud_recognition_model.pth", weights_only=True))
model.eval()
model.to(device)

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 初始化 Flask 应用
app = Flask(__name__)

# 定义类别名称
classes = ["cirrus", "cirrostratus", "cirrocumulus", "altocumulus", "altostratus", "cumulus", "cumulonimbus",
           "nimbostratus", "stratocumulus", "stratus", "contrail"]


# 创建图片上传接口
@app.route("/predict", methods=["POST"])
def predict():
    # 确保文件存在
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # 打开图片并进行预处理
        image = Image.open(file.stream).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        # 进行预测
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
        print(predicted)
        print(outputs)
        # 返回结果
        return jsonify({'class': classes[predicted.item()]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
