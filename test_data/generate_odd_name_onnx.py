# This script generates the .onnx file with a bunch of different special chars
# in the filename. It takes a 1x2 uint32 tensor and produces a 1x1-element
# uint32 output containing the sum of the 2 inputs.
import torch

class AddModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs.sum(1).int()

def main():
    model = AddModel()
    model.eval()
    x = torch.ones((1, 2), dtype=torch.int32)
    file_name = "example ż 大 김.onnx"
    torch.onnx.export(model, (x,), file_name, input_names=["in"],
        output_names=["out"])
    print(f"{file_name} saved OK.")

if __name__ == "__main__":
    main()

