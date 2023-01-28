# This script sets up and "trains" a toy pytorch network, that trains a NN to
# map a 1x4 vector to a 1x2 vector containing [sum, max difference] of the
# input values. Finally, it exports the network to an ONNX file to use in
# testing.
import torch
from torch.nn.functional import relu
import json

def fake_dataset(size):
    """ Returns a dataset filled with our fake training data. """
    inputs = torch.rand((size, 1, 4))
    outputs = torch.zeros((size, 1, 2))
    for i in range(size):
        outputs[i][0][0] = inputs[i][0].sum()
        outputs[i][0][1] = inputs[i][0].max() - inputs[i][0].min()
    return torch.utils.data.TensorDataset(inputs, outputs)

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
        batch_size = len(data)
        conv_out = relu(self.conv(data))
        conv_flattened = torch.flatten(conv_out, start_dim=1)
        data_flattened = torch.flatten(data, start_dim=1)
        combined = torch.cat((conv_flattened, data_flattened), dim=1)
        output = relu(self.fc(combined))
        output = output.view(batch_size, 1, 2)
        return output

def get_test_loss(model, loader, loss_function):
    """ Just runs a single epoch of data from the given loader. Returns the
    average loss per batch. The provided model is expected to be in eval mode.
    """
    i = 0
    total_loss = 0.0
    for in_data, desired_result in loader:
        produced_result = model(in_data)
        loss = loss_function(desired_result, produced_result)
        total_loss += loss.item()
        i += 1
    return total_loss / i

def save_model(model, output_filename):
    """ Saves the model to an onnx file with the given name. Assumes the model
    is in eval mode. """
    print("Saving network to " + output_filename)
    dummy_input = torch.rand(1, 1, 4)
    input_names = ["1x4 Input Vector"]
    output_names = ["1x2 Output Vector"]
    torch.onnx.export(model, dummy_input, output_filename,
        input_names=input_names, output_names=output_names)
    return None

def print_sample(model):
    """ Prints a sample input and output computation using the model. Expects
    the model to be in eval mode. """
    example_input = torch.rand(1, 1, 4)
    result = model(example_input)
    print("Sample model execution:")
    print("    Example input: " + str(example_input))
    print("    Produced output: " + str(result))
    return None

def save_sample_json(model, output_name):
    """ Saves a JSON file containing an input and an output from the network,
    for use when validating execution of the ONNX network. """
    example_input = torch.rand(1, 1, 4)
    result = model(example_input)
    json_content = {}
    json_content["input_shape"] = list(example_input.shape)
    json_content["flattened_input"] = list(example_input.flatten().tolist())
    json_content["output_shape"] = list(result.shape)
    json_content["flattened_output"] = list(result.flatten().tolist())
    with open(output_name, "w") as f:
        json.dump(json_content, f, indent="  ")
    return None

def main():
    print("Generating train and test datasets...")
    train_data = fake_dataset(100 * 1000)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
        batch_size=16, shuffle=True)
    test_data = fake_dataset(10 * 1000)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
        batch_size=16)
    model = SumAndDiffModel()
    model.train()
    loss_function = torch.nn.L1Loss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    for epoch in range(8):
        i = 0
        total_loss = 0.0
        for in_data, desired_result in train_loader:
            i += 1
            produced_result = model(in_data)
            loss = loss_function(desired_result, produced_result)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            if (i % 1000) == 1:
                print("Epoch %d, iteration %d. Current loss = %f" % (epoch, i,
                    loss.item()))
        train_loss = total_loss / i
        print("  => Average train-set loss: " + str(train_loss))
        model.eval()
        with torch.no_grad():
            test_loss = get_test_loss(model, test_loader, loss_function)
        model.train()
        print("  => Average test-set loss: " + str(test_loss))

    model.eval()
    with torch.no_grad():
        save_model(model, "example_network.onnx")
        save_sample_json(model, "example_network_results.json")
        print_sample(model)
    print("Done!")

if __name__ == "__main__":
    main()

