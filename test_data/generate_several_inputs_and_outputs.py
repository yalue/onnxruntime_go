# This script creates example_several_inputs_and_outputs.onnx to use in
# testing. The "network" is entirely deterministic, and is intended just to
# illustrate a wide variety of inputs and outputs with varying names,
# dimensions, and types.
#
# Inputs:
#  - "input 1": a 2x5x2x5 int32 tensor
#  - "input 2": a 2x3x20 float tensor
#  - "input 3": a 9-element bfloat16 tensor
#
# Outputs:
#  - "output 1": A 10x10 element int64 tensor
#  - "output 2": A 1x2x3x4x5 element double tensor
#
# The contents of the inputs and outputs are arbitrary.
import torch

class ManyInputOutputModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, c):
        output_a = a.reshape((10, 10))
        output_a = output_a.type(torch.int64)
        output_b = b.reshape((1, 2, 3, 4, 5))
        output_b = output_b.type(torch.double)
        # Just to make sure we use input C.
        output_a[0][0] += c[0].type(torch.int64)
        return output_a, output_b

def main():
    model = ManyInputOutputModel()
    model.eval()
    out_name = "example_several_inputs_and_outputs.onnx"
    input_a = torch.zeros((2, 5, 2, 5), dtype=torch.int32)
    input_b = torch.zeros((2, 3, 20), dtype=torch.float)
    input_c = torch.zeros((9), dtype=torch.bfloat16)
    torch.onnx.export(model, (input_a, input_b, input_c), out_name,
        input_names = ["input 1", "input 2", "input 3"],
        output_names = ["output 1", "output 2"])
    print(f"{out_name} saved OK.")

if __name__ == "__main__":
    main()

