# This script creates example_multitype.onnx to use in testing.
# The "network" doesn't actually do much other than cast around some types and
# perform basic arithmetic.  It takes two inputs:
#  - "InputA": A 1x1 8-bit unsigned int tensor
#  - "InputB": A 2x2 64-bit float tensor
#
# It produces 2 outputs:
#  - "OutputA": A 2x2 16-bit signed int tensor, equal to (InputB * InputA) - 512
#  - "OutputB": A 1x1 64-bit int tensor, equal to InputA multiplied by 1234
import torch

class DifferentTypesModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_a, input_b):
        output_a = input_b * input_a[0][0][0]
        output_a -= 512
        output_a = output_a.type(torch.int16)
        output_b = input_a.type(torch.int64)
        output_b *= 1234
        return output_a, output_b

def fake_inputs():
    input_a = torch.rand((1, 1, 1)) * 255.0
    input_a = input_a.type(torch.uint8)
    input_b = torch.rand((1, 2, 2), dtype=torch.float64)
    return input_a, input_b


def main():
    model = DifferentTypesModel()
    model.eval()
    input_a, input_b = fake_inputs()
    output_a, output_b = model(input_a, input_b)
    print(f"Example inputs: A = {input_a!s}, B = {input_b!s}")
    print(f"Produced outputs: A = {output_a!s}, B = {output_b!s}")

    out_name = "example_multitype.onnx"
    print(f"Saving model as {out_name}")
    input_names = ["InputA", "InputB"]
    output_names = ["OutputA", "OutputB"]
    torch.onnx.export(model, (input_a, input_b), out_name,
        input_names=input_names, output_names=output_names)
    print(f"{out_name} saved OK.")

if __name__ == "__main__":
    main()

