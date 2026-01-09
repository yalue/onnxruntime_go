# This file generates example_complex_numbers.onnx. At the time of writing,
# there is almost no support whatsoever in onnxruntime for complex tensors,
# despite the type technically being supported. Therefore, we're limited to the
# identity operator to verify that they work.

import numpy as np
import onnx
from onnx import helper, TensorProto

# Build ONNX model: two Identity ops, one for complex64, one for complex128
in1 = helper.make_tensor_value_info('in_c64',  TensorProto.COMPLEX64,  [None])
in2 = helper.make_tensor_value_info('in_c128', TensorProto.COMPLEX128, [None])
out1 = helper.make_tensor_value_info('out_c64',  TensorProto.COMPLEX64,  [None])
out2 = helper.make_tensor_value_info('out_c128', TensorProto.COMPLEX128, [None])

n1 = helper.make_node('Identity', ['in_c64'],  ['out_c64'])
n2 = helper.make_node('Identity', ['in_c128'], ['out_c128'])

filename = "example_complex_numbers.onnx"
graph = helper.make_graph([n1, n2], 'ComplexIdentity', [in1, in2], [out1, out2])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
onnx.save(model, filename)
print(f"Saved {filename} OK.")

