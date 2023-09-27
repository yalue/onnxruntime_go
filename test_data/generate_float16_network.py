# This script creates example_float16.onnx to use in testing.
# It takes one input:
#  - "InputA": A 2x2x2 16-bit float16 tensor
# It produces one output:
#  - "OutputA": A 2x2x2 16-bit bfloat16 tensor
#
# The "network" just multiplies each element in the input by 3.0
import torch

class Float16Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_a):
        output_a = input_a * 3.0
        output_a = output_a.type(torch.bfloat16)
        return output_a

def fake_inputs():
    return torch.rand((1, 2, 2, 2), dtype=torch.float16)

def main():
    model = Float16Model()
    model.eval()
    input_a = torch.rand((1, 2, 2, 2), dtype=torch.float16)
    output_a = model(input_a)

    out_name = "example_float16.onnx"
    torch.onnx.export(model, (input_a), out_name, input_names=["InputA"],
        output_names=["OutputA"])
    print(f"{out_name} saved OK.")

if __name__ == "__main__":
    main()

