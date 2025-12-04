# This script generates "example_strings.onnx". This example takes a 1xN tensor
# of N strings, and produces two 1xN outputs: one with the strings converted to
# lowercase and one with the strings converted to uppercase.
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

def main():
    # Describe the inputs and outputs
    input_info = helper.make_tensor_value_info("input", TensorProto.STRING,
        [None])
    output_lower_info = helper.make_tensor_value_info("output_lower",
        TensorProto.STRING, [None])
    output_upper_info = helper.make_tensor_value_info("output_upper",
        TensorProto.STRING, [None])

    node_lower = helper.make_node(
        "StringNormalizer",
        inputs=["input"],
        outputs=["output_lower"],
        case_change_action="LOWER",
    )

    node_upper = helper.make_node(
        "StringNormalizer",
        inputs=["input"],
        outputs=["output_upper"],
        case_change_action="UPPER",
    )

    graph = helper.make_graph(
        [node_lower, node_upper],
        "strings_example_graph",
        [input_info],
        [output_lower_info, output_upper_info],
    )

    model = helper.make_model(graph,
        producer_name="generate_strings_example.py")
    onnx.checker.check_model(model)
    filename = "example_strings.onnx"
    onnx.save(model, filename)
    print(f"Saved {filename} OK. Testing...")

    session = ort.InferenceSession(filename)

    inputs = np.array(["I", "eAt", "POTATOEs!!"])

    output_lower, output_upper = session.run(None, {"input": inputs})

    print("Inputs: " + str(inputs))
    print("Lowercase: " + str(output_lower))
    print("Upercase: " + str(output_upper))

if __name__ == "__main__":
    main()

