package onnxruntime_go

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"testing"
)

// This type is read from JSON and used to determine the inputs and expected
// outputs for an ONNX network.
type testInputsInfo struct {
	InputShape      []int64   `json:"input_shape"`
	FlattenedInput  []float32 `json:"flattened_input"`
	OutputShape     []int64   `json:"output_shape"`
	FlattenedOutput []float32 `json:"flattened_output"`
}

// This must be called prior to running each test.
func InitializeRuntime(t *testing.T) {
	if IsInitialized() {
		return
	}
	if runtime.GOOS == "windows" {
		SetSharedLibraryPath("test_data/onnxruntime.dll")
	} else {
		if runtime.GOARCH == "arm64" {
			SetSharedLibraryPath("test_data/onnxruntime_arm64.so")
		} else {
			SetSharedLibraryPath("test_data/onnxruntime.so")
		}
	}
	e := InitializeEnvironment()
	if e != nil {
		t.Logf("Failed setting up onnxruntime environment: %s\n", e)
		t.FailNow()
	}
}

// Used to obtain the shape
func parseInputsJSON(path string, t *testing.T) *testInputsInfo {
	toReturn := testInputsInfo{}
	f, e := os.Open(path)
	if e != nil {
		t.Logf("Failed opening %s: %s\n", path, e)
		t.FailNow()
	}
	defer f.Close()
	d := json.NewDecoder(f)
	e = d.Decode(&toReturn)
	if e != nil {
		t.Logf("Failed decoding %s: %s\n", path, e)
		t.FailNow()
	}
	return &toReturn
}

// Returns an error if any element between a and b don't match.
func floatsEqual(a, b []float32) error {
	if len(a) != len(b) {
		return fmt.Errorf("Length mismatch: %d vs %d", len(a), len(b))
	}
	for i := range a {
		diff := a[i] - b[i]
		if diff < 0 {
			diff = -diff
			// Arbitrarily chosen precision.
			if diff >= 0.00000001 {
				return fmt.Errorf("Data element %d doesn't match: %f vs %v",
					i, a[i], b[i])
			}
		}
	}
	return nil
}

func TestTensorTypes(t *testing.T) {
	// It would be nice to compare this, but doing that would require exposing
	// the underlying C types in Go; the testing package doesn't support cgo.
	type myFloat float64
	dataType := GetTensorElementDataType[myFloat]()
	t.Logf("Got data type for float64-based double: %d\n", dataType)
}

func TestCreateTensor(t *testing.T) {
	InitializeRuntime(t)
	defer DestroyEnvironment()
	s := NewShape(1, 2, 3)
	tensor1, e := NewEmptyTensor[uint8](s)
	if e != nil {
		t.Logf("Failed creating %s uint8 tensor: %s\n", s, e)
		t.FailNow()
	}
	defer tensor1.Destroy()
	if len(tensor1.GetData()) != 6 {
		t.Logf("Incorrect data length for tensor1: %d\n",
			len(tensor1.GetData()))
	}
	// Make sure that the underlying tensor created a copy of the shape we
	// passed to NewEmptyTensor.
	s[1] = 3
	if tensor1.GetShape()[1] == s[1] {
		t.Logf("Modifying the original shape incorrectly changed the " +
			"tensor's shape.\n")
		t.FailNow()
	}

	// Try making a tensor with a different data type.
	s = NewShape(2, 5)
	data := []float32{1.0}
	_, e = NewTensor(s, data)
	if e == nil {
		t.Logf("Didn't get error when creating a tensor with too little " +
			"data.\n")
		t.FailNow()
	}
	t.Logf("Got expected error when creating a tensor without enough data: "+
		"%s\n", e)

	// It shouldn't be an error to create a tensor with too *much* underlying
	// data; we'll just use the first portion of it.
	data = []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
	tensor2, e := NewTensor(s, data)
	if e != nil {
		t.Logf("Error creating tensor with data: %s\n", e)
		t.FailNow()
	}
	defer tensor2.Destroy()
	// Make sure the tensor's internal slice only refers to the part we care
	// about, and not the entire slice.
	if len(tensor2.GetData()) != 10 {
		t.Logf("New tensor data contains %d elements, when it should "+
			"contain 10.\n", len(tensor2.GetData()))
		t.FailNow()
	}
}

func TestCloneTensor(t *testing.T) {
	InitializeRuntime(t)
	originalData := []float32{1, 2, 3, 4}
	originalTensor, e := NewTensor(NewShape(2, 2), originalData)
	if e != nil {
		t.Logf("Error creating tensor: %s\n", e)
		t.FailNow()
	}
	clone, e := originalTensor.Clone()
	if e != nil {
		t.Logf("Error cloning tensor: %s\n", e)
		t.FailNow()
	}
	if !clone.GetShape().Equals(originalTensor.GetShape()) {
		t.Logf("Clone shape (%s) doesn't match original shape (%s)\n",
			clone.GetShape(), originalTensor.GetShape())
		t.FailNow()
	}
	cloneData := clone.GetData()
	for i := range originalData {
		if cloneData[i] != originalData[i] {
			t.Logf("Clone data incorrect at index %d: %f (expected %f)\n",
				i, cloneData[i], originalData[i])
			t.FailNow()
		}
	}
	cloneData[2] = 1337
	if originalData[2] != 3 {
		t.Logf("Modifying clone data effected the original.\n")
		t.FailNow()
	}
}

func TestExampleNetwork(t *testing.T) {
	InitializeRuntime(t)
	defer func() {
		e := DestroyEnvironment()
		if e != nil {
			t.Logf("Error cleaning up environment: %s\n", e)
			t.FailNow()
		}
	}()

	// Create input and output tensors
	inputs := parseInputsJSON("test_data/example_network_results.json", t)
	inputTensor, e := NewTensor(Shape(inputs.InputShape),
		inputs.FlattenedInput)
	if e != nil {
		t.Logf("Failed creating input tensor: %s\n", e)
		t.FailNow()
	}
	defer inputTensor.Destroy()
	outputTensor, e := NewEmptyTensor[float32](Shape(inputs.OutputShape))
	if e != nil {
		t.Logf("Failed creating output tensor: %s\n", e)
		t.FailNow()
	}
	defer outputTensor.Destroy()

	// Set up and run the session.
	session, e := NewSession[float32]("test_data/example_network.onnx",
		[]string{"1x4 Input Vector"}, []string{"1x2 Output Vector"},
		[]*Tensor[float32]{inputTensor}, []*Tensor[float32]{outputTensor})
	if e != nil {
		t.Logf("Failed creating session: %s\n", e)
		t.FailNow()
	}
	defer session.Destroy()
	e = session.Run()
	if e != nil {
		t.Logf("Failed to run the session: %s\n", e)
		t.FailNow()
	}
	e = floatsEqual(outputTensor.GetData(), inputs.FlattenedOutput)
	if e != nil {
		t.Logf("The neural network didn't produce the correct result: %s\n", e)
		t.FailNow()
	}
}
