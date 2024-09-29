import os
import torch
from flask import Flask, request, jsonify
from torchvision import transforms
from PIL import Image
from torchvision import models

current_dir = os.path.dirname(os.path.abspath(__file__))

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet101()
model.fc = torch.nn.Linear(model.fc.in_features, 21)
model.load_state_dict(
    torch.load(os.path.join(current_dir, "models", "flower_recognition_model_best.pth"), map_location=device,
               weights_only=True
               ), strict=False)
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

classes = ["astilbe", "bellflower", "black_eyed_susan", "calendula", "california_poppy", "carnation", "common_daisy",
           "coreopsis",
           "dandelion", "garden_roses", "gardenias", "hibiscus", "hydrangeas", "iris", "lilies", "orchids", "peonies",
           "rose", "sunflower", "tulip", "water_lily"]


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
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = probabilities.argmax(dim=1).item()
            print(outputs, "outputs")
            print(probabilities, "probabilities")
            print(predicted_class, "predicted_class")

        #     _, predicted = torch.max(outputs, 1)
        # print(predicted, "predicted")
        # print(predicted.item(), "predicted.item")
        # print(outputs, "outputs")
        # 返回结果
        return jsonify({'class': classes[predicted_class], 'probability': probabilities[0][predicted_class].item()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
