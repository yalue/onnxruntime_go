# This file is used to modify the metadata of example_big_compute.onnx for the
# sake of testing. The choice of example_big_compute.onnx was arbitrary.
import onnx

def main():
    file_path = "example_big_compute.onnx"
    model = onnx.load(file_path)
    model.doc_string = "This is a test description."
    model.model_version = 1337
    model.domain = "test domain"
    custom_1 = model.metadata_props.add()
    custom_1.key = "test key 1"
    custom_1.value = ""
    custom_2 = model.metadata_props.add()
    custom_2.key = "test key 2"
    custom_2.value = "Test key 2 value"
    onnx.save(model, file_path)

if __name__ == "__main__":
    main()
