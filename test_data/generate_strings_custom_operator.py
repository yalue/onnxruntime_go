# This script generates an .onnx file that uses a custom op provided by
# onnxruntime-extensions (ai.onnx.contrib::StringUpper). The model takes a 1-D
# tensor of strings and outputs the uppercased version of each element.
#
# Running this model with plain onnxruntime (no extensions registered) will
# fail, because StringUpper is not part of the standard ONNX op set.
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
from onnxruntime_extensions import get_library_path

# Custom ops from onnxruntime-extensions live in this domain.
EXTENSIONS_DOMAIN = "ai.onnx.contrib"

def main():
    # Describe the inputs and outputs
    in_tensor = helper.make_tensor_value_info(
        "input", TensorProto.STRING, [None])
    out_tensor = helper.make_tensor_value_info(
        "output", TensorProto.STRING, [None])

    # Single node calling the extensions-provided StringUpper op.
    upper_node = helper.make_node(
        op_type="StringUpper",
        inputs=["input"],
        outputs=["output"],
        domain=EXTENSIONS_DOMAIN,
        name="upper",
    )

    graph = helper.make_graph(
        nodes=[upper_node],
        name="needs_extensions_graph",
        inputs=[in_tensor],
        outputs=[out_tensor],
    )

    # Declare the opset imports: standard ONNX + the extensions domain.
    opset_imports = [
        helper.make_opsetid("", 17),                 # default ONNX domain
        helper.make_opsetid(EXTENSIONS_DOMAIN, 1),   # onnxruntime-extensions
    ]

    model = helper.make_model(
        graph,
        producer_name="needs_extensions_generator",
        opset_imports=opset_imports,
    )

    file_name = "example_needs_extensions.onnx"
    onnx.save(model, file_name)
    print(f"{file_name} saved OK.")

    so = ort.SessionOptions()
    so.register_custom_ops_library(get_library_path())


    session = ort.InferenceSession(file_name, so)

    inputs = np.array(["I", "eAt", "PoTAtOEs!!"])
    outputs = session.run(None, {"input": inputs})
    output_lower = outputs[0]

    print("Inputs:  " + str(inputs))
    print("Outputs: " + str(output_lower))

if __name__ == "__main__":
    main()
