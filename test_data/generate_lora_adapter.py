# This script generates "example_lora.onnx" and "example_lora.onnx_adapter",
# used when testing LoraAdapter support. The network computes
# in @ base_weight + (in @ lora_a) @ lora_b, where lora_a and lora_b are
# inputs backed by zero-size default initializers, so running the network
# without overriding them only produces the base term. The .onnx_adapter file
# overrides lora_a and lora_b with 4x1 and 1x4 matrices, changing the outputs.
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

def make_network():
    input_info = helper.make_tensor_value_info("in", TensorProto.FLOAT,
        [1, 4])
    lora_a_info = helper.make_tensor_value_info("lora_a", TensorProto.FLOAT,
        [4, "lora_rank"])
    lora_b_info = helper.make_tensor_value_info("lora_b", TensorProto.FLOAT,
        ["lora_rank", 4])
    output_info = helper.make_tensor_value_info("out", TensorProto.FLOAT,
        [1, 4])

    base_weight = helper.make_tensor("base_weight", TensorProto.FLOAT,
        [4, 4], np.arange(1, 17, dtype=np.float32).flatten())
    # The defaults for the LoRA parameters have a rank of 0, so their
    # contribution to the output is a 1x4 matrix of zeros.
    default_a = helper.make_tensor("lora_a", TensorProto.FLOAT, [4, 0], [])
    default_b = helper.make_tensor("lora_b", TensorProto.FLOAT, [0, 4], [])

    base_node = helper.make_node("MatMul", ["in", "base_weight"],
        ["base_out"])
    lora_node_a = helper.make_node("MatMul", ["in", "lora_a"], ["a_out"])
    lora_node_b = helper.make_node("MatMul", ["a_out", "lora_b"], ["b_out"])
    add_node = helper.make_node("Add", ["base_out", "b_out"], ["out"])

    graph = helper.make_graph(
        [base_node, lora_node_a, lora_node_b, add_node],
        "lora_example_graph",
        [input_info, lora_a_info, lora_b_info],
        [output_info],
        initializer=[base_weight, default_a, default_b],
    )
    model = helper.make_model(graph,
        opset_imports=[helper.make_opsetid("", 21)])
    onnx.checker.check_model(model)
    output_name = "example_lora.onnx"
    onnx.save_model(model, output_name)
    print(f"Saved {output_name} OK.")

def make_adapter():
    lora_a = np.array([3, 4, 5, 6], dtype=np.float32).reshape(4, 1)
    lora_b = np.array([7, 8, 9, 10], dtype=np.float32).reshape(1, 4)
    parameters = {
        "lora_a": ort.OrtValue.ortvalue_from_numpy(lora_a),
        "lora_b": ort.OrtValue.ortvalue_from_numpy(lora_b),
    }
    adapter = ort.AdapterFormat()
    adapter.set_adapter_version(1)
    adapter.set_model_version(1)
    adapter.set_parameters(parameters)
    output_name = "example_lora.onnx_adapter"
    adapter.export_adapter(output_name)
    print(f"Saved {output_name} OK.")

def main():
    make_network()
    make_adapter()

if __name__ == "__main__":
    main()
