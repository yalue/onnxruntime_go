# This script creates example_dynamic_sizes.py to use in testing. It takes a
# batch of [-1, 10] input vectors and produces [-1] output scalars---the sum of
# each input vector (where -1 is a dynamic batch size).
import torch

class DynamicSizeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_batch):
        return input_batch.sum(1)

def main():
    model = DynamicSizeModel()
    model.eval()
    test_input = torch.rand((123, 10), dtype=torch.float32)
    dynamic_axes = {
        "input_vectors": [0],
        "output_scalars": [0],
    }
    output_name = "example_dynamic_axes.onnx"
    torch.onnx.export(model, (test_input), output_name,
        input_names=["input_vectors"], output_names=["output_scalars"],
        dynamic_axes=dynamic_axes)
    print(f"Saved {output_name} OK.")

if __name__ == "__main__":
    main()

