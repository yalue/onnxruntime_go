# This script creates example_big_fanout.onnx to use in testing. The idea is
# to create a newtwork where parallelism makes a big difference.
import torch

class BigFanoutModel(torch.nn.Module):
    """ Maps a 1x4 vector to another 1x4 vector, but goes through a large
    number of parallelizable useless FC operations. """
    def __init__(self):
        super().__init__()
        self.fanout_amount = 100
        self.matrices = [torch.rand((4, 4)) for i in range(self.fanout_amount)]

    def forward(self, x):
        # Do fanout_amount matrix multiplies, then merge and sum the result.
        tmp_results = [
            torch.matmul(x, self.matrices[i])
            for i in range(self.fanout_amount)
        ]
        combined_tensor = torch.cat(tmp_results)
        return combined_tensor.sum(0)

def main():
    model = BigFanoutModel()
    model.float()
    model.eval()
    test_input = torch.rand((1, 4), dtype=torch.float32)
    output_name = "example_big_fanout.onnx"
    test_input.requires_grad = False
    for param in model.parameters():
        param.requires_grad = False
    with torch.no_grad():
        torch.onnx.export(model, (test_input), output_name,
            input_names=["input"], output_names=["output"])
    print(f"Saved {output_name} OK.")

if __name__ == "__main__":
    main()

