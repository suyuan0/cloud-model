import torch
import torch.nn as nn
from torchvision import models
import os

class FlowerClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FlowerClassifier, self).__init__()
        self.model = models.resnet101()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

current_dir = os.path.dirname(os.path.abspath(__file__))

num_classes = 21
image_size = 224

model_path = os.path.join(current_dir, "models/flower/2024-09-26", "flower_recognition_model.pth")
onnx_path = os.path.join(current_dir, "", "flower_model.onnx")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FlowerClassifier(num_classes)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

# 创建一个示例输入张量（假设输入大小为 (1, 3, 224, 224)）
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

print(f"Model saved to {onnx_path}")