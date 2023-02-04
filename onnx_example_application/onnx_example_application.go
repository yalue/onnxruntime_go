// This application loads a test ONNX network and executes it on some fixed
// data. It serves as an example of how to use the onnxruntime wrapper library.
package main

import (
	"encoding/json"
	"fmt"
	"github.com/yalue/onnxruntime"
	"os"
	"runtime"
)

// This type is read from JSON and used to determine the inputs and expected
// outputs for an ONNX network.
type testInputsInfo struct {
	InputShape      []int64   `json:"input_shape"`
	FlattenedInput  []float32 `json:"flattened_input"`
	OutputShape     []int64   `json:"output_shape"`
	FlattenedOutput []float32 `json:"flattened_output"`
}

// Loads JSON that contains the shapes and data used by the test ONNX network.
// Requires the path to the JSON file.
func loadInputsJSON(path string) (*testInputsInfo, error) {
	f, e := os.Open(path)
	if e != nil {
		return nil, fmt.Errorf("Error opening %s: %w", path, e)
	}
	defer f.Close()
	d := json.NewDecoder(f)
	var toReturn testInputsInfo
	e = d.Decode(&toReturn)
	if e != nil {
		return nil, fmt.Errorf("Error decoding %s: %w", path, e)
	}
	return &toReturn, nil
}

func run() int {
	if runtime.GOOS == "windows" {
		onnxruntime.SetSharedLibraryPath("../test_data/onnxruntime.dll")
	} else {
		if runtime.GOARCH == "arm64" {
			onnxruntime.SetSharedLibraryPath("../test_data/onnxruntime_arm64.so")
		} else {
			onnxruntime.SetSharedLibraryPath("../test_data/onnxruntime.so")
		}
	}
	e := onnxruntime.InitializeEnvironment()
	if e != nil {
		fmt.Printf("Error initializing the onnxruntime environment: %s\n", e)
		return 1
	}
	fmt.Printf("The onnxruntime environment initialized OK.\n")

	// Load the JSON with the test input and output data.
	testInputs, e := loadInputsJSON(
		"../test_data/example_network_results.json")
	if e != nil {
		fmt.Printf("Error reading example inputs from JSON: %s\n", e)
		return 1
	}

	// Create the session with the test onnx network
	session, e := onnxruntime.NewSimpleSession[float32](
		"../test_data/example_network.onnx")
	if e != nil {
		fmt.Printf("Error initializing the ONNX session: %s\n", e)
		return 1
	}
	defer session.Destroy()

	// Create input and output tensors.
	inputShape := onnxruntime.Shape(testInputs.InputShape)
	outputShape := onnxruntime.Shape(testInputs.OutputShape)
	inputTensor, e := onnxruntime.NewTensor(inputShape,
		testInputs.FlattenedInput)
	if e != nil {
		fmt.Printf("Failed getting input tensor: %s\n", e)
		return 1
	}
	defer inputTensor.Destroy()
	outputTensor, e := onnxruntime.NewEmptyTensor[float32](outputShape)
	if e != nil {
		fmt.Printf("Failed creating output tensor: %s\n", e)
		return 1
	}
	defer outputTensor.Destroy()

	// Actually run the network.
	e = session.SimpleRun(inputTensor, outputTensor)
	if e != nil {
		fmt.Printf("Failed running network: %s\n", e)
		return 1
	}

	for i := range outputTensor.GetData() {
		fmt.Printf("Output value %d: expected %f, got %f\n", i,
			outputTensor.GetData()[i], testInputs.FlattenedOutput[i])
	}

	// Ordinarily, it is probably fine to call this using defer, but we do it
	// here just so we can print a status message after the cleanup completes.
	e = onnxruntime.CleanupEnvironment()
	if e != nil {
		fmt.Printf("Error cleaning up the environment: %s\n", e)
		return 1
	}
	fmt.Printf("The onnxruntime environment was cleaned up OK.\n")
	return 0
}

func main() {
	os.Exit(run())
}
