# This script creates example_big_compute.onnx to use in testing.
# The "network" is entirely deterministic; it simply does a large amount of
# hopefully expensive arithmetic operations.
#
# It takes one input: "Input", a one-dimensional vector of 1024*1024*50 32-bit
# floats, and produces one output, named "Output" of the same dimensions.
import torch

class BigComputeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        for i in range(40):
            x = x / 10.0
            x = x * 10.0
        return x

def main():
    model = BigComputeModel()
    model.eval()
    x = torch.zeros((1, 1024 * 1024 * 50), dtype=torch.float32)

    out_name = "example_big_compute.onnx"
    torch.onnx.export(model, x, out_name,
        input_names=["Input"], output_names=["Output"])
    print(f"{out_name} saved OK.")

if __name__ == "__main__":
    main()

