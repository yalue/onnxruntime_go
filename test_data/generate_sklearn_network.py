# This script is a modified version of the example from
# https://pypi.org/project/skl2onnx/, which we use to produce
# sklearn_randomforest.onnx. sklearn makes heavy use of onnxruntime maps and
# sequences in its networks, so this is used for testing those data types.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
inputs, outputs = iris.data, iris.target
inputs = inputs.astype(np.float32)
inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs, outputs)
classifier = RandomForestClassifier()
classifier.fit(inputs_train, outputs_train)

# Convert into ONNX format.
from skl2onnx import to_onnx
output_filename = "sklearn_randomforest.onnx"
onnx_content = to_onnx(classifier, inputs[:1])
with open(output_filename, "wb") as f:
    f.write(onnx_content.SerializeToString())

# Compute the prediction with onnxruntime.
import onnxruntime as ort

def float_formatter(f):
    return f"{float(f):.06f}"

np.set_printoptions(formatter = {'float_kind': float_formatter})
session = ort.InferenceSession(output_filename)
print(f"Input names: {[n.name for n in session.get_inputs()]!s}")
print(f"Output names: {[o.name for o in session.get_outputs()]!s}")
example_inputs = inputs_test.astype(np.float32)[:6]
print(f"Inputs shape = {example_inputs.shape!s}")
onnx_predictions = session.run(["output_label", "output_probability"],
    {"X": example_inputs})
labels = onnx_predictions[0]
probabilities = onnx_predictions[1]

print(f"Inputs to network: {example_inputs.astype(np.float32)}")
print(f"ONNX predicted labels: {labels!s}")
print(f"ONNX predicted probabilities: {probabilities!s}")

