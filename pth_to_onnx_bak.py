import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet101_Weights
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

num_classes = 21
image_size = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(
    torch.load(os.path.join(current_dir, "models", "flower_recognition_model.pth"), map_location=device,
               weights_only=True))
model = model.to(device)
model.eval()


def export_onnx_model(model, input_size=(1, 3, image_size, image_size), model_path=os.path.join(current_dir, "", "flower_model.onnx")):
    dummy_input = torch.randn(input_size).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=11
    )
    print(f"ONNX 模型已导出到 {model_path}")

export_onnx_model(model)
