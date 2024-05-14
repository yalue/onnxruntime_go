import torch
from torch.nn.functional import relu
from pathlib import Path
import onnx
import onnxruntime.training.artifacts as artifacts

class SumAndDiffModel(torch.nn.Module):
    """ Just a standard, fairly minimal, pytorch model for generating the NN.
    """
    def __init__(self):
        super().__init__()
        # We'll do four 1x4 convolutions to make the network more interesting.
        self.conv = torch.nn.Conv1d(1, 4, 4)
        # We'll follow the conv with a FC layer to produce the outputs. The
        # input to the FC layer are the 4 conv outputs concatenated with the
        # original input.
        self.fc = torch.nn.Linear(8, 2)

    def forward(self, data):
        batch_size = data.shape[0]
        conv_out = relu(self.conv(data))
        conv_flattened = torch.flatten(conv_out, start_dim=1)
        data_flattened = torch.flatten(data, start_dim=1)
        combined = torch.cat((conv_flattened, data_flattened), dim=1)
        output = relu(self.fc(combined))
        output = output.view(batch_size, 1, 2)
        return output
    
def main():
    model = SumAndDiffModel()

    # Export the model to ONNX.
    training_artifacts_path = Path(".", "training_test")
    training_artifacts_path.mkdir(exist_ok=True, parents=True)
    model_name = "training_network"
    torch.onnx.export(model, torch.zeros(10, 1, 4),
                Path(".", "training_test", f"{model_name}.onnx").__str__(),
                input_names=["input"], output_names=["output"])

    # Load the onnx model and generate artifacts
    onnx_model = onnx.load(Path("training_test", "training_network.onnx"))
    requires_grad = ["conv.weight", "conv.bias", "fc.weight", "fc.bias"]
    model.train()

    # Generate the training artifacts.
    artifacts.generate_artifacts(
    onnx_model,
    requires_grad=requires_grad,
    frozen_params=[],
    loss=artifacts.LossType.L1Loss,
    optimizer=artifacts.OptimType.AdamW,
    artifact_directory=Path(".", "training_test"))

if __name__ == "__main__":
    main()