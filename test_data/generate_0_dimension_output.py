# This script creates example_0_dim_output.onnx to use in testing. The idea is
# that the network produces an output with one a dimension of size 0.
import torch

class ZeroDimOutputModel(torch.nn.Module):
    """ Takes a 2x8 input, and produces a 2xNx8 output, where N is the sum of
    the first input column, cast to an int. In tests, the input will be all 0s,
    so this should result in a 2x0x8 output. """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        tmp = x.sum(0)
        return tmp.unsqueeze(0).expand(2, tmp.int()[0], -1)

def main():
    model = ZeroDimOutputModel()
    model.eval()
    x = torch.ones((2, 8), dtype=torch.float32)
    out_name = "example_0_dim_output.onnx"
    torch.onnx.export(model, (x,), out_name, input_names=["x"],
        output_names=["y"])
    print(f"{out_name} saved OK.")

if __name__ == "__main__":
    main()

