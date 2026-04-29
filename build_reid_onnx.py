"""
Build a ResNet50-based ReID ONNX model exactly matching the architecture
expected by ReIDExtractor in reid_test.py:
  - Input:  (1, 3, 256, 128)  — BGR crop normalised to ImageNet stats
  - Output: (1, 2048)         — L2-normalised embedding vector
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50
import os

class ResNet50ReID(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet50(weights=None)
        # Remove final FC + avgpool, replace with adaptive pool → 2048-d feature
        self.backbone = nn.Sequential(*list(base.children())[:-2])  # up to layer4
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))
        # Lightweight BN-head (standard in ReID literature)
        self.bn       = nn.BatchNorm1d(2048)
        self.bn.bias.requires_grad_(False)

    def forward(self, x):
        x = self.backbone(x)           # (B, 2048, 8, 4) for 256×128 input
        x = self.pool(x).flatten(1)    # (B, 2048)
        x = self.bn(x)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x

model = ResNet50ReID()
model.eval()

dummy = torch.zeros(1, 3, 256, 128)
with torch.no_grad():
    out = model(dummy)
    print(f"Output shape: {out.shape}")   # should be (1, 2048)
    print(f"Output norm:  {out.norm():.4f}")  # should be ~1.0

os.makedirs("onnx_model", exist_ok=True)
torch.onnx.export(
    model, dummy,
    "onnx_model/resnet50_reid.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17,
)
print("Exported → onnx_model/resnet50_reid.onnx")
