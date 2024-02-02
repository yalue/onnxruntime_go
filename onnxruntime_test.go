package onnxruntime_go

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"testing"
)

// Always use the same RNG seed for benchmarks, so we can compare the
// performance on the same random input data.
const benchmarkRNGSeed = 12345678

// This type is read from JSON and used to determine the inputs and expected
// outputs for an ONNX network.
type testInputsInfo struct {
	InputShape      []int64   `json:"input_shape"`
	FlattenedInput  []float32 `json:"flattened_input"`
	OutputShape     []int64   `json:"output_shape"`
	FlattenedOutput []float32 `json:"flattened_output"`
}

// If the ONNXRUNTIME_SHARED_LIBRARY_PATH environment variable is set, then
// we'll try to use its contents as the location of the shared library for
// these tests. Otherwise, we'll fall back to trying the shared library copies
// in the test_data directory.
func getTestSharedLibraryPath(t testing.TB) string {
	toReturn := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH")
	if toReturn != "" {
		return toReturn
	}
	if runtime.GOOS == "windows" {
		return "test_data/onnxruntime.dll"
	}
	if runtime.GOARCH == "arm64" {
		if runtime.GOOS == "darwin" {
			return "test_data/onnxruntime_arm64.dylib"
		}
		return "test_data/onnxruntime_arm64.so"
	}
	return "test_data/onnxruntime.so"
}

// This must be called prior to running each test.
func InitializeRuntime(t testing.TB) {
	if IsInitialized() {
		return
	}
	SetSharedLibraryPath(getTestSharedLibraryPath(t))
	e := InitializeEnvironment()
	if e != nil {
		t.Logf("Failed setting up onnxruntime environment: %s\n", e)
		t.FailNow()
	}
}

// Should be called at the end of each test to de-initialize the runtime.
func CleanupRuntime(t testing.TB) {
	e := DestroyEnvironment()
	if e != nil {
		t.Logf("Error cleaning up environment: %s\n", e)
		t.FailNow()
	}
}

// Used to obtain the shape
func parseInputsJSON(path string, t testing.TB) *testInputsInfo {
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
		}
		// Arbitrarily chosen precision.
		if diff >= 0.00000001 {
			return fmt.Errorf("Data element %d doesn't match: %f vs %v",
				i, a[i], b[i])
		}
	}
	return nil
}

// Returns an empty tensor with the given type and shape, or fails the test on
// error.
func newTestTensor[T TensorData](t testing.TB, s Shape) *Tensor[T] {
	toReturn, e := NewEmptyTensor[T](s)
	if e != nil {
		t.Logf("Failed creating empty tensor with shape %s: %s\n", s, e)
		t.FailNow()
	}
	return toReturn
}

func TestTensorTypes(t *testing.T) {
	type myFloat float64
	dataType := TensorElementDataType(GetTensorElementDataType[myFloat]())
	expected := TensorElementDataType(TensorElementDataTypeDouble)
	if dataType != expected {
		t.Logf("Expected float64 data type to be %d (%s), got %d (%s)\n",
			expected, expected, dataType, dataType)
		t.FailNow()
	}
	t.Logf("Got data type for float64-based double: %d (%s)\n",
		dataType, dataType)
}

func TestCreateTensor(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)
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

func TestBadTensorShapes(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)
	s := NewShape()
	_, e := NewEmptyTensor[float64](s)
	if e == nil {
		t.Logf("Didn't get an error when creating a tensor with an empty " +
			"shape.\n")
		t.FailNow()
	}
	t.Logf("Got expected error when creating a tensor with an empty shape: "+
		"%s\n", e)
	s = NewShape(10, 0, 10)
	_, e = NewEmptyTensor[uint16](s)
	if e == nil {
		t.Logf("Didn't get an error when creating a tensor with a shape " +
			"containing a 0 dimension.\n")
		t.FailNow()
	}
	t.Logf("Got expected error when creating a tensor with a 0 dimension: "+
		"%s\n", e)
	s = NewShape(10, 10, -10)
	_, e = NewEmptyTensor[int32](s)
	if e == nil {
		t.Logf("Didn't get an error when creating a tensor with a negative " +
			"dimension.\n")
		t.FailNow()
	}
	t.Logf("Got expected error when creating a tensor with a negative "+
		"dimension: %s\n", e)
	s = NewShape(10, -10, -10)
	_, e = NewEmptyTensor[uint64](s)
	if e == nil {
		t.Logf("Didn't get an error when creating a tensor with two " +
			"negative dimensions.\n")
		t.FailNow()
	}
	t.Logf("Got expected error when creating a tensor with two negative "+
		"dimensions: %s\n", e)
	s = NewShape(int64(1)<<62, 1, int64(1)<<62)
	_, e = NewEmptyTensor[float32](s)
	if e == nil {
		t.Logf("Didn't get an error when creating a tensor with an " +
			"overflowing shape.\n")
		t.FailNow()
	}
	t.Logf("Got expected error when creating a tensor with an overflowing "+
		"shape: %s\n", e)
}

func TestCloneTensor(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)
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

func TestZeroTensorContents(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)
	a := newTestTensor[float64](t, NewShape(3, 4, 5))
	defer a.Destroy()
	data := a.GetData()
	for i := range data {
		data[i] = float64(i)
	}
	t.Logf("Before zeroing: a[%d] = %f\n", len(data)-1, data[len(data)-1])
	a.ZeroContents()
	for i, v := range data {
		if v != 0.0 {
			t.Logf("a[%d] = %f, expected it to be set to 0.\n", i, v)
			t.FailNow()
		}
	}

	// Do the same basic test with a CustomDataTensor
	shape := NewShape(2, 3, 4, 5)
	customData := randomBytes(123, 2*shape.FlattenedSize())
	b, e := NewCustomDataTensor(shape, customData, TensorElementDataTypeUint16)
	if e != nil {
		t.Logf("Error creating custom data tensor: %s\n", e)
		t.FailNow()
	}
	defer b.Destroy()
	for i := range customData {
		// This will wrap around, but doesn't matter. We just need arbitrary
		// nonzero data for the test.
		customData[i] = uint8(i)
	}
	t.Logf("Start of custom data before zeroing: % x\n", customData[0:10])
	b.ZeroContents()
	for i, v := range customData {
		if v != 0 {
			t.Logf("b[%d] = %d, expected it to be set to 0.\n", i, v)
			t.FailNow()
		}
	}
}

func TestExampleNetwork(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)

	// Create input and output tensors
	inputs := parseInputsJSON("test_data/example_network_results.json", t)
	inputTensor, e := NewTensor(Shape(inputs.InputShape),
		inputs.FlattenedInput)
	if e != nil {
		t.Logf("Failed creating input tensor: %s\n", e)
		t.FailNow()
	}
	defer inputTensor.Destroy()
	outputTensor := newTestTensor[float32](t, Shape(inputs.OutputShape))
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

func TestExampleNetworkDynamic(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)

	// Create input and output tensors
	inputs := parseInputsJSON("test_data/example_network_results.json", t)
	inputTensor, e := NewTensor(Shape(inputs.InputShape),
		inputs.FlattenedInput)
	if e != nil {
		t.Logf("Failed creating input tensor: %s\n", e)
		t.FailNow()
	}
	defer inputTensor.Destroy()
	outputTensor := newTestTensor[float32](t, Shape(inputs.OutputShape))
	defer outputTensor.Destroy()

	// Set up and run the session without specifying the inputs and outputs shapes
	session, e := NewDynamicSession[float32, float32]("test_data/example_network.onnx",
		[]string{"1x4 Input Vector"}, []string{"1x2 Output Vector"})
	if e != nil {
		t.Logf("Failed creating session: %s\n", e)
		t.FailNow()
	}
	defer session.Destroy()
	// running with the input
	e = session.Run([]*Tensor[float32]{inputTensor}, []*Tensor[float32]{outputTensor})
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

func TestEnableDisableTelemetry(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)

	e := EnableTelemetry()
	if e != nil {
		t.Logf("Error enabling onnxruntime telemetry: %s\n", e)
		t.Fail()
	}
	e = DisableTelemetry()
	if e != nil {
		t.Logf("Error disabling onnxruntime telemetry: %s\n", e)
		t.Fail()
	}
	e = EnableTelemetry()
	if e != nil {
		t.Logf("Error re-enabling onnxruntime telemetry after disabling: %s\n",
			e)
		t.Fail()
	}
}

func TestArbitraryTensors(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)

	tensorShape := NewShape(2, 2)
	tensorA, e := NewTensor(tensorShape, []uint8{1, 2, 3, 4})
	if e != nil {
		t.Logf("Error creating uint8 tensor: %s\n", e)
		t.FailNow()
	}
	defer tensorA.Destroy()
	tensorB, e := NewTensor(tensorShape, []float64{5, 6, 7, 8})
	if e != nil {
		t.Logf("Error creating float64 tensor: %s\n", e)
		t.FailNow()
	}
	defer tensorB.Destroy()
	tensorC, e := NewTensor(tensorShape, []int16{9, 10, 11, 12})
	if e != nil {
		t.Logf("Error creating int16 tensor: %s\n", e)
		t.FailNow()
	}
	defer tensorC.Destroy()
	tensorList := []ArbitraryTensor{tensorA, tensorB, tensorC}
	for i, v := range tensorList {
		ortValue := v.GetInternals().ortValue
		t.Logf("ArbitraryTensor %d: Data type %d, shape %s, OrtValue %p\n",
			i, v.DataType(), v.GetShape(), ortValue)
	}
}

// Used for testing the operation of test_data/example_multitype.onnx
func randomMultitypeInputs(t *testing.T, seed int64) (*Tensor[uint8],
	*Tensor[float64]) {
	rng := rand.New(rand.NewSource(seed))
	inputA := newTestTensor[uint8](t, NewShape(1, 1, 1))
	// We won't use newTestTensor here, otherwise we won't have a chance to
	// destroy inputA on failure.
	inputB, e := NewEmptyTensor[float64](NewShape(1, 2, 2))
	if e != nil {
		inputA.Destroy()
		t.Logf("Failed creating input B: %s\n", e)
		t.FailNow()
	}
	inputA.GetData()[0] = uint8(rng.Intn(256))
	for i := 0; i < 4; i++ {
		inputB.GetData()[i] = rng.Float64()
	}
	return inputA, inputB
}

// Used when checking the output produced by test_data/example_multitype.onnx
func getExpectedMultitypeOutputs(inputA *Tensor[uint8],
	inputB *Tensor[float64]) ([]int16, []int64) {
	outputA := make([]int16, 4)
	dataA := inputA.GetData()[0]
	dataB := inputB.GetData()
	for i := 0; i < len(outputA); i++ {
		outputA[i] = int16((dataB[i] * float64(dataA)) - 512)
	}
	return outputA, []int64{int64(dataA) * 1234}
}

// Verifies that the given tensor's data matches the expected content. Prints
// an error and fails the test if anything doesn't match.
func verifyTensorData[T TensorData](t *testing.T, tensor *Tensor[T],
	expectedContent []T) {
	data := tensor.GetData()
	if len(data) != len(expectedContent) {
		t.Logf("Expected tensor to contain %d elements, but it contains %d.\n",
			len(expectedContent), len(data))
		t.FailNow()
	}
	for i, v := range expectedContent {
		if v != data[i] {
			t.Logf("Data mismatch at element index %d: expected %v, got %v\n",
				i, v, data[i])
			t.FailNow()
		}
	}
}

// Tests a session taking multiple input tensors of different types and
// producing multiple output tensors of different types.
func TestDifferentInputOutputTypes(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)

	inputA, inputB := randomMultitypeInputs(t, 9999)
	defer inputA.Destroy()
	defer inputB.Destroy()
	outputA := newTestTensor[int16](t, NewShape(1, 2, 2))
	defer outputA.Destroy()
	outputB := newTestTensor[int64](t, NewShape(1, 1, 1))
	defer outputB.Destroy()

	session, e := NewAdvancedSession("test_data/example_multitype.onnx",
		[]string{"InputA", "InputB"}, []string{"OutputA", "OutputB"},
		[]ArbitraryTensor{inputA, inputB},
		[]ArbitraryTensor{outputA, outputB}, nil)
	if e != nil {
		t.Logf("Failed creating session: %s\n", e)
		t.FailNow()
	}
	defer session.Destroy()
	e = session.Run()
	if e != nil {
		t.Logf("Error running session: %s\n", e)
		t.FailNow()
	}
	expectedA, expectedB := getExpectedMultitypeOutputs(inputA, inputB)
	verifyTensorData(t, outputA, expectedA)
	verifyTensorData(t, outputB, expectedB)
}

func TestDynamicDifferentInputOutputTypes(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)

	session, e := NewDynamicAdvancedSession("test_data/example_multitype.onnx",
		[]string{"InputA", "InputB"}, []string{"OutputA", "OutputB"}, nil)
	defer session.Destroy()

	numTests := 100
	aInputs := make([]*Tensor[uint8], numTests)
	bInputs := make([]*Tensor[float64], numTests)
	aOutputs := make([]*Tensor[int16], numTests)
	bOutputs := make([]*Tensor[int64], numTests)

	// Make sure we clean up all the tensors created for this test, even if we
	// somehow fail before we've created them all.
	defer func() {
		for i := 0; i < numTests; i++ {
			if aInputs[i] != nil {
				aInputs[i].Destroy()
			}
			if bInputs[i] != nil {
				bInputs[i].Destroy()
			}
			if aOutputs[i] != nil {
				aOutputs[i].Destroy()
			}
			if bOutputs[i] != nil {
				bOutputs[i].Destroy()
			}
		}
	}()

	// Actually create the inputs and run the tests.
	for i := 0; i < numTests; i++ {
		aInputs[i], bInputs[i] = randomMultitypeInputs(t, 999+int64(i))
		aOutputs[i] = newTestTensor[int16](t, NewShape(1, 2, 2))
		bOutputs[i] = newTestTensor[int64](t, NewShape(1, 1, 1))
		e = session.Run([]ArbitraryTensor{aInputs[i], bInputs[i]},
			[]ArbitraryTensor{aOutputs[i], bOutputs[i]})
		if e != nil {
			t.Logf("Failed running session for test %d: %s\n", i, e)
			t.FailNow()
		}
	}

	// Now that all the tests ran, check the outputs. If the
	// DynamicAdvancedSession worked properly, each run should have only
	// modified its given outputs.
	for i := 0; i < numTests; i++ {
		expectedA, expectedB := getExpectedMultitypeOutputs(aInputs[i],
			bInputs[i])
		verifyTensorData(t, aOutputs[i], expectedA)
		verifyTensorData(t, bOutputs[i], expectedB)
	}
}

func TestDynamicAllocatedOutputTensor(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)

	session, err := NewDynamicAdvancedSession("test_data/example_multitype.onnx",
		[]string{"InputA", "InputB"}, []string{"OutputA", "OutputB"}, nil)
	defer session.Destroy()

	// Actually create the inputs and run the tests.
	aInput, bInput := randomMultitypeInputs(t, 999)
	var outputs [2]ArbitraryTensor
	err = session.Run([]ArbitraryTensor{aInput, bInput}, outputs[:])
	if err != nil {
		t.Logf("Failed running session: %s\n", err)
		t.FailNow()
	}

	expectedA, expectedB := getExpectedMultitypeOutputs(aInput, bInput)
	if outputA, ok := outputs[0].(*Tensor[int16]); !ok {
		t.Logf("Expected outputA to be of type %T, got of type %T\n", outputA, outputs[0])
		t.FailNow()
	} else if expectedShape := NewShape(1, 2, 2); !outputA.shape.Equals(expectedShape) {
		t.Logf("Expected outputA to be of shape %s, got of shape %s\n", expectedShape, outputA.shape)
		t.FailNow()
	} else {
		verifyTensorData(t, outputA, expectedA)
	}
	if outputB, ok := outputs[1].(*Tensor[int64]); !ok {
		t.Logf("Expected outputB to be of type %T, got of type %T\n", outputB, outputs[1])
		t.FailNow()
	} else if expectedShape := NewShape(1, 1, 1); !outputB.shape.Equals(expectedShape) {
		t.Logf("Expected outputB to be of shape %s, got of shape %s\n", expectedShape, outputB.shape)
		t.FailNow()
	} else {
		verifyTensorData(t, outputB, expectedB)
	}

}

func TestWrongInputs(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)

	session, e := NewDynamicAdvancedSession("test_data/example_multitype.onnx",
		[]string{"InputA", "InputB"}, []string{"OutputA", "OutputB"}, nil)
	defer session.Destroy()

	inputA, inputB := randomMultitypeInputs(t, 123456)
	defer inputA.Destroy()
	defer inputB.Destroy()
	outputA := newTestTensor[int16](t, NewShape(1, 2, 2))
	defer outputA.Destroy()
	outputB := newTestTensor[int64](t, NewShape(1, 1, 1))
	defer outputB.Destroy()

	// Make sure that passing a tensor with the wrong type but correct shape
	// will correctly cause an error rather than a crash, whether used as an
	// input or output.
	wrongTypeTensor := newTestTensor[float32](t, NewShape(1, 2, 2))
	defer wrongTypeTensor.Destroy()
	e = session.Run([]ArbitraryTensor{inputA, inputB},
		[]ArbitraryTensor{wrongTypeTensor, outputB})
	if e == nil {
		t.Logf("Didn't get expected error when passing a float32 tensor in " +
			"place of an int16 output tensor.\n")
		t.FailNow()
	}
	t.Logf("Got expected error when passing a float32 tensor in place of an "+
		"int16 output tensor: %s\n", e)
	e = session.Run([]ArbitraryTensor{inputA, wrongTypeTensor},
		[]ArbitraryTensor{outputA, outputB})
	if e == nil {
		t.Logf("Didn't get expected error when passing a float32 tensor in " +
			"place of a float64 input tensor.\n")
		t.FailNow()
	}
	t.Logf("Got expected error when passing a float32 tensor in place of a "+
		"float64 input tensor: %s\n", e)

	// Make sure that passing a tensor with the wrong shape but correct type
	// will cause an error rather than a crash, when using as an input or an
	// output.
	wrongShapeInput := newTestTensor[uint8](t, NewShape(22))
	defer wrongShapeInput.Destroy()
	e = session.Run([]ArbitraryTensor{wrongShapeInput, inputB},
		[]ArbitraryTensor{outputA, outputB})
	if e == nil {
		t.Logf("Didn't get expected error when running with an incorrectly " +
			"shaped input.\n")
		t.FailNow()
	}
	t.Logf("Got expected error when running with an incorrectly shaped "+
		"input: %s\n", e)
	wrongShapeOutput := newTestTensor[int64](t, NewShape(1, 1, 1, 1, 1, 1))
	defer wrongShapeOutput.Destroy()
	e = session.Run([]ArbitraryTensor{inputA, inputB},
		[]ArbitraryTensor{outputA, wrongShapeOutput})
	if e == nil {
		t.Logf("Didn't get expected error when running with an incorrectly " +
			"shaped output.\n")
		t.FailNow()
	}
	t.Logf("Got expected error when running with an incorrectly shaped "+
		"output: %s\n", e)

	e = session.Run([]ArbitraryTensor{inputA, inputB},
		[]ArbitraryTensor{outputA, outputB})
	if e != nil {
		t.Logf("Got error attempting to (correctly) Run a session after "+
			"attempting to use incorrect inputs or outputs: %s\n", e)
		t.FailNow()
	}
}

func TestGetInputOutputInfo(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)
	file := "test_data/example_several_inputs_and_outputs.onnx"
	inputs, outputs, e := GetInputOutputInfo(file)
	if e != nil {
		t.Logf("Error getting input and output info for %s: %s\n", file, e)
		t.FailNow()
	}
	if len(inputs) != 3 {
		t.Logf("Expected 3 inputs, got %d\n", len(inputs))
		t.FailNow()
	}
	if len(outputs) != 2 {
		t.Logf("Expected 2 outputs, got %d\n", len(outputs))
		t.FailNow()
	}
	for i, v := range inputs {
		t.Logf("Input %d: %s\n", i, &v)
	}
	for i, v := range outputs {
		t.Logf("Output %d: %s\n", i, &v)
	}

	if outputs[1].Name != "output 2" {
		t.Logf("Incorrect output 1 name: %s, expected \"output 2\"\n",
			outputs[1].Name)
		t.Fail()
	}
	expectedShape := NewShape(1, 2, 3, 4, 5)
	if !outputs[1].Dimensions.Equals(expectedShape) {
		t.Logf("Incorrect output 1 shape: %s, expected %s\n",
			outputs[1].Dimensions, expectedShape)
		t.Fail()
	}
	var expectedType TensorElementDataType = TensorElementDataTypeDouble
	if outputs[1].DataType != expectedType {
		t.Logf("Incorrect output 1 data type: %s, expected %s\n",
			outputs[1].DataType, expectedType)
		t.Fail()
	}
	if inputs[0].Name != "input 1" {
		t.Logf("Incorrect input 0 name: %s, expected \"input 1\"\n",
			inputs[0].Name)
		t.Fail()
	}
	expectedShape = NewShape(2, 5, 2, 5)
	if !inputs[0].Dimensions.Equals(expectedShape) {
		t.Logf("Incorrect input 0 shape: %s, expected %s\n",
			inputs[0].Dimensions, expectedShape)
		t.Fail()
	}
	expectedType = TensorElementDataTypeInt32
	if inputs[0].DataType != expectedType {
		t.Logf("Incorrect input 0 data type: %s, expected %s\n",
			inputs[0].DataType, expectedType)
		t.Fail()
	}
}

func TestModelMetadata(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)
	file := "test_data/example_network.onnx"
	metadata, e := GetModelMetadata(file)
	if e != nil {
		t.Logf("Error getting metadata for %s: %s\n", file, e)
		t.FailNow()
	}
	// We'll just test Destroy once; after this we won't check its return value
	e = metadata.Destroy()
	if e != nil {
		t.Logf("Error destroying metadata: %s\n", e)
		t.FailNow()
	}

	// Try getting the metadata from a session instead of from a file.
	// NOTE: All of the expected values here were manually set using the
	// test_data/modify_metadata.py script after generating the network. See
	// that script for the expected values of each of the metadata accesors.
	file = "test_data/example_big_compute.onnx"
	session, e := NewDynamicAdvancedSession(file, []string{"Input"},
		[]string{"Output"}, nil)
	if e != nil {
		t.Logf("Error creating session: %s\n", e)
		t.FailNow()
	}
	defer session.Destroy()
	metadata, e = session.GetModelMetadata()
	if e != nil {
		t.Logf("Error getting metadata from DynamicAdvancedSession: %s\n", e)
		t.FailNow()
	}
	defer metadata.Destroy()
	producerName, e := metadata.GetProducerName()
	if e != nil {
		t.Logf("Error getting producer name: %s\n", e)
		t.Fail()
	} else {
		t.Logf("Got producer name: %s\n", producerName)
	}
	graphName, e := metadata.GetGraphName()
	if e != nil {
		t.Logf("Error getting graph name: %s\n", e)
		t.Fail()
	} else {
		t.Logf("Got graph name: %s\n", graphName)
	}
	domainStr, e := metadata.GetDomain()
	if e != nil {
		t.Logf("Error getting domain: %s\n", e)
		t.Fail()
	} else {
		t.Logf("Got domain: %s\n", domainStr)
		if domainStr != "test domain" {
			t.Logf("Incorrect domain string, expected \"test domain\"\n")
			t.Fail()
		}
	}
	description, e := metadata.GetDescription()
	if e != nil {
		t.Logf("Error getting description: %s\n", e)
		t.Fail()
	} else {
		t.Logf("Got description: %s\n", description)
	}
	version, e := metadata.GetVersion()
	if e != nil {
		t.Logf("Error getting version: %s\n", e)
		t.Fail()
	} else {
		t.Logf("Got version: %d\n", version)
		if version != 1337 {
			t.Logf("Incorrect version number, expected 1337\n")
			t.Fail()
		}
	}
	mapKeys, e := metadata.GetCustomMetadataMapKeys()
	if e != nil {
		t.Logf("Error getting custom metadata keys: %s\n", e)
		t.FailNow()
	}
	t.Logf("Got %d custom metadata map keys.\n", len(mapKeys))
	if len(mapKeys) != 2 {
		t.Logf("Incorrect number of custom metadata keys, expected 2")
		t.Fail()
	}
	for _, k := range mapKeys {
		value, present, e := metadata.LookupCustomMetadataMap(k)
		if e != nil {
			t.Logf("Error looking up key %s in custom metadata: %s\n", k, e)
			t.Fail()
		} else {
			if !present {
				t.Logf("LookupCustomMetadataMap didn't return true for a " +
					"key that should be present in the map\n")
				t.Fail()
			}
			t.Logf("  Metadata key \"%s\" = \"%s\"\n", k, value)
		}
	}
	badValue, present, e := metadata.LookupCustomMetadataMap("invalid key")
	if len(badValue) != 0 {
		t.Logf("Didn't get an empty string when looking up an invalid "+
			"metadata key, got \"%s\" instead\n", badValue)
		t.FailNow()
	}
	if present {
		t.Logf("LookupCustomMetadataMap didn't return false for a key that " +
			"isn't in the map\n")
		t.Fail()
	}
	// Tossing in this check, since the docs aren't clear on this topic. (The
	// docs specify returning an empty string, but do not mention a non-NULL
	// OrtStatus.) At the time of writing, it does _not_ return an error.
	if e == nil {
		t.Logf("Informational: looking up an invalid metadata key doesn't " +
			"return an error\n")
	} else {
		t.Logf("Informational: got error when looking up an invalid "+
			"metadata key: %s\n", e)
	}
}

func randomBytes(seed, n int64) []byte {
	toReturn := make([]byte, n)
	rng := rand.New(rand.NewSource(seed))
	rng.Read(toReturn)
	return toReturn
}

func TestCustomDataTensors(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)
	shape := NewShape(2, 3, 4, 5)
	tensorData := randomBytes(123, 2*shape.FlattenedSize())
	// This could have been created using a Tensor[uint16], but we'll make sure
	// it works this way, too.
	v, e := NewCustomDataTensor(shape, tensorData, TensorElementDataTypeUint16)
	if e != nil {
		t.Logf("Error creating uint16 CustomDataTensor: %s\n", e)
		t.FailNow()
	}
	shape[0] = 6
	if v.GetShape().Equals(shape) {
		t.Logf("CustomDataTensor didn't correctly create a Clone of its shape")
		t.FailNow()
	}
	e = v.Destroy()
	if e != nil {
		t.Logf("Error destroying CustomDataTensor: %s\n", e)
		t.FailNow()
	}
	tensorData = randomBytes(1234, 2*shape.FlattenedSize())
	v, e = NewCustomDataTensor(shape, tensorData, TensorElementDataTypeFloat16)
	if e != nil {
		t.Logf("Error creating float16 tensor: %s\n", e)
		t.FailNow()
	}
	e = v.Destroy()
	if e != nil {
		t.Logf("Error destroying float16 tensor: %s\n", e)
		t.FailNow()
	}
	// Make sure we don't fail if providing more data than necessary
	shape[0] = 1
	v, e = NewCustomDataTensor(shape, tensorData,
		TensorElementDataTypeBFloat16)
	if e != nil {
		t.Logf("Got error when creating a tensor with more data than "+
			"necessary: %s\n", e)
		t.FailNow()
	}
	v.Destroy()

	// Make sure we fail when using a bad shape
	shape = NewShape(0, -1, -2)
	v, e = NewCustomDataTensor(shape, tensorData, TensorElementDataTypeFloat16)
	if e == nil {
		v.Destroy()
		t.Logf("Didn't get error when creating custom tensor with an " +
			"invalid shape\n")
		t.FailNow()
	}
	t.Logf("Got expected error creating tensor with invalid shape: %s\n", e)
	shape = NewShape(1, 2, 3, 4, 5)
	tensorData = []byte{1, 2, 3, 4}
	v, e = NewCustomDataTensor(shape, tensorData, TensorElementDataTypeUint8)
	if e == nil {
		v.Destroy()
		t.Logf("Didn't get error when creating custom tensor with too " +
			"little data\n")
		t.FailNow()
	}
	t.Logf("Got expected error when creating custom data tensor with "+
		"too little data: %s\n", e)

	// Make sure we fail when using a bad type
	tensorData = []byte{1, 2, 3, 4, 5, 6, 7, 8}
	badType := TensorElementDataType(0xffffff)
	v, e = NewCustomDataTensor(NewShape(2), tensorData, badType)
	if e == nil {
		v.Destroy()
		t.Logf("Didn't get error when creating custom tensor with bad type\n")
		t.FailNow()
	}
	t.Logf("Got expected error when creating custom data tensor with bad "+
		"type: %s\n", e)
}

// Converts a slice of floats to their representation as bfloat16 bytes.
func floatsToBfloat16(f []float32) []byte {
	toReturn := make([]byte, 2*len(f))
	// bfloat16 is just a truncated version of a float32
	for i := range f {
		bf16Bits := uint16(math.Float32bits(f[i]) >> 16)
		toReturn[i*2] = uint8(bf16Bits)
		toReturn[i*2+1] = uint8(bf16Bits >> 8)
	}
	return toReturn
}

func TestFloat16Network(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)

	// The network takes a 1x2x2x2 float16 input
	inputData := []byte{
		// 0.0, 1.0, 2.0, 3.0
		0x00, 0x00, 0x00, 0x3c, 0x00, 0x40, 0x00, 0x42,
		// 4.0, 5.0, 6.0, 7.0
		0x00, 0x44, 0x00, 0x45, 0x00, 0x46, 0x00, 0x47,
	}
	// The network produces a 1x2x2x2 bfloat16 output: the input multiplied
	// by 3
	expectedOutput := floatsToBfloat16([]float32{0, 3, 6, 9, 12, 15, 18, 21})
	outputData := make([]byte, len(expectedOutput))
	inputTensor, e := NewCustomDataTensor(NewShape(1, 2, 2, 2), inputData,
		TensorElementDataTypeFloat16)
	if e != nil {
		t.Logf("Error creating input tensor: %s\n", e)
		t.FailNow()
	}
	defer inputTensor.Destroy()
	outputTensor, e := NewCustomDataTensor(NewShape(1, 2, 2, 2), outputData,
		TensorElementDataTypeBFloat16)
	if e != nil {
		t.Logf("Error creating output tensor: %s\n", e)
		t.FailNow()
	}
	defer outputTensor.Destroy()

	session, e := NewAdvancedSession("test_data/example_float16.onnx",
		[]string{"InputA"}, []string{"OutputA"},
		[]ArbitraryTensor{inputTensor}, []ArbitraryTensor{outputTensor}, nil)
	if e != nil {
		t.Logf("Error creating session: %s\n", e)
		t.FailNow()
	}
	defer session.Destroy()
	e = session.Run()
	if e != nil {
		t.Logf("Error running session: %s\n", e)
		t.FailNow()
	}
	for i := range outputData {
		if outputData[i] != expectedOutput[i] {
			t.Logf("Incorrect output byte at index %d: 0x%02x (expected "+
				"0x%02x)\n", i, outputData[i], expectedOutput[i])
			t.FailNow()
		}
	}
}

// See the comment in generate_network_big_compute.py for information about
// the inputs and outputs used for testing or benchmarking session options.
func prepareBenchmarkTensors(t testing.TB, seed int64) (*Tensor[float32],
	*Tensor[float32]) {
	vectorLength := int64(1024 * 1024 * 50)
	inputData := make([]float32, vectorLength)
	rng := rand.New(rand.NewSource(seed))
	for i := range inputData {
		inputData[i] = rng.Float32()
	}
	input, e := NewTensor(NewShape(1, vectorLength), inputData)
	if e != nil {
		t.Logf("Error creating input tensor: %s\n", e)
		t.FailNow()
	}
	output, e := NewEmptyTensor[float32](NewShape(1, vectorLength))
	if e != nil {
		input.Destroy()
		t.Logf("Error creating output tensor: %s\n", e)
		t.FailNow()
	}
	return input, output
}

// Used mostly when testing different execution providers. Runs the
// example_big_compute.onnx network on a session created with the given
// options. May fail or skip the test on error. The runtime must have already
// been initialized when calling this.
func testBigSessionWithOptions(t *testing.T, options *SessionOptions) {
	input, output := prepareBenchmarkTensors(t, 1337)
	defer input.Destroy()
	defer output.Destroy()
	session, e := NewAdvancedSession("test_data/example_big_compute.onnx",
		[]string{"Input"}, []string{"Output"}, []ArbitraryTensor{input},
		[]ArbitraryTensor{output}, options)
	if e != nil {
		t.Logf("Error creating session: %s\n", e)
		t.FailNow()
	}
	defer session.Destroy()
	e = session.Run()
	if e != nil {
		t.Logf("Error running the session: %s\n", e)
		t.FailNow()
	}
}

// Used when benchmarking different execution providers. Otherwise, basically
// identical in usage to testBigSessionWithOptions.
func benchmarkBigSessionWithOptions(b *testing.B, options *SessionOptions) {
	// It's also OK for the caller to have already stopped the timer, but we'll
	// make sure it's stopped here.
	b.StopTimer()
	input, output := prepareBenchmarkTensors(b, benchmarkRNGSeed)
	defer input.Destroy()
	defer output.Destroy()
	session, e := NewAdvancedSession("test_data/example_big_compute.onnx",
		[]string{"Input"}, []string{"Output"}, []ArbitraryTensor{input},
		[]ArbitraryTensor{output}, options)
	if e != nil {
		b.Logf("Error creating session: %s\n", e)
		b.FailNow()
	}
	defer session.Destroy()
	b.StartTimer()
	for n := 0; n < b.N; n++ {
		e = session.Run()
		if e != nil {
			b.Logf("Error running iteration %d/%d: %s\n", n+1, b.N, e)
			b.FailNow()
		}
	}
}

func TestSessionOptions(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)
	options, e := NewSessionOptions()
	if e != nil {
		t.Logf("Error creating session options: %s\n", e)
		t.FailNow()
	}
	defer options.Destroy()
	e = options.SetIntraOpNumThreads(3)
	if e != nil {
		t.Logf("Error setting intra-op num threads: %s\n", e)
		t.FailNow()
	}
	e = options.SetInterOpNumThreads(1)
	if e != nil {
		t.Logf("Error setting inter-op num threads: %s\n", e)
		t.FailNow()
	}
	e = options.SetCpuMemArena(true)
	if e != nil {
		t.Logf("Error setting CPU memory arena: %s\n", e)
		t.FailNow()
	}
	e = options.SetMemPattern(true)
	if e != nil {
		t.Logf("Error setting memory pattern: %s\n", e)
		t.FailNow()
	}
	testBigSessionWithOptions(t, options)
}

// Very similar to TestSessionOptions, but structured as a benchmark.
func runNumThreadsBenchmark(b *testing.B, nThreads int) {
	// Don't run the benchmark timer when doing initialization.
	b.StopTimer()
	InitializeRuntime(b)
	defer CleanupRuntime(b)
	options, e := NewSessionOptions()
	if e != nil {
		b.Logf("Error creating options: %s\n", e)
		b.FailNow()
	}
	defer options.Destroy()
	e = options.SetIntraOpNumThreads(nThreads)
	if e != nil {
		b.Logf("Error setting intra-op threads to %d: %s\n", nThreads, e)
		b.FailNow()
	}
	e = options.SetInterOpNumThreads(nThreads)
	if e != nil {
		b.Logf("Error setting inter-op threads to %d: %s\n", nThreads, e)
		b.FailNow()
	}
	benchmarkBigSessionWithOptions(b, options)
}

func BenchmarkOpSingleThreaded(b *testing.B) {
	runNumThreadsBenchmark(b, 1)
}

func BenchmarkOpMultiThreaded(b *testing.B) {
	runNumThreadsBenchmark(b, 0)
}

// Creates a SessionOptions struct that's configured to enable CUDA. Skips the
// test if CUDA isn't supported. If some other error occurs, this will fail the
// test instead. There may be other possible places for failures to occur due
// to CUDA not being supported, or incorrectly configured, but this at least
// checks for the ones I've encountered on my system.
func getCUDASessionOptions(t testing.TB) *SessionOptions {
	// First, create the CUDA options
	cudaOptions, e := NewCUDAProviderOptions()
	if e != nil {
		// This is where things seem to fail if the onnxruntime library version
		// doesn't support CUDA.
		t.Skipf("Error creating CUDA provider options: %s. "+
			"Your version of the onnxruntime library may not support CUDA. "+
			"Skipping the remainder of this test.\n", e)
	}
	defer cudaOptions.Destroy()
	e = cudaOptions.Update(map[string]string{"device_id": "0"})
	if e != nil {
		// This is where things seem to fail if the system doesn't support CUDA
		// or if CUDA is misconfigured somehow (i.e. a wrong version that isn't
		// supported by onnxruntime, libraries not being located correctly,
		// etc.)
		t.Skipf("Error updating CUDA options to use device ID 0: %s. "+
			"Your system may not support CUDA, or CUDA may be misconfigured "+
			"or a version incompatible with this version of onnxruntime. "+
			"Skipping the remainder of this test.\n", e)
	}
	// Next, provide the CUDA options to the sesison options
	sessionOptions, e := NewSessionOptions()
	if e != nil {
		t.Logf("Error creating SessionOptions: %s\n", e)
		t.FailNow()
	}
	e = sessionOptions.AppendExecutionProviderCUDA(cudaOptions)
	if e != nil {
		sessionOptions.Destroy()
		t.Logf("Error setting CUDA execution provider options: %s\n", e)
		t.FailNow()
	}
	return sessionOptions
}

func TestCUDASession(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)
	sessionOptions := getCUDASessionOptions(t)
	defer sessionOptions.Destroy()
	testBigSessionWithOptions(t, sessionOptions)
}

func BenchmarkCUDASession(b *testing.B) {
	b.StopTimer()
	InitializeRuntime(b)
	defer CleanupRuntime(b)
	sessionOptions := getCUDASessionOptions(b)
	defer sessionOptions.Destroy()
	benchmarkBigSessionWithOptions(b, sessionOptions)
}

// Creates a SessionOptions struct that's configured to enable TensorRT.
// Basically the same as getCUDASessionOptions; see the comments there.
func getTensorRTSessionOptions(t testing.TB) *SessionOptions {
	trtOptions, e := NewTensorRTProviderOptions()
	if e != nil {
		t.Skipf("Error creating TensorRT provider options; %s. "+
			"Your version of the onnxruntime library may not include "+
			"TensorRT support. Skipping the remainder of this test.\n", e)
	}
	defer trtOptions.Destroy()
	// Arbitrarily update an option to test trtOptions.Update()
	e = trtOptions.Update(
		map[string]string{"trt_max_partition_iterations": "60"})
	if e != nil {
		t.Skipf("Error updating TensorRT options: %s. Your system may not "+
			"support TensorRT, TensorRT may be misconfigured, or it may be "+
			"incompatible with this build of onnxruntime. Skipping the "+
			"remainder of this test.\n", e)
	}
	sessionOptions, e := NewSessionOptions()
	if e != nil {
		t.Logf("Error creating SessionOptions: %s\n", e)
		t.FailNow()
	}
	e = sessionOptions.AppendExecutionProviderTensorRT(trtOptions)
	if e != nil {
		sessionOptions.Destroy()
		t.Logf("Error setting TensorRT execution provider: %s\n", e)
		t.FailNow()
	}
	return sessionOptions
}

func TestTensorRTSession(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)
	sessionOptions := getTensorRTSessionOptions(t)
	defer sessionOptions.Destroy()
	testBigSessionWithOptions(t, sessionOptions)
}

func BenchmarkTensorRTSession(b *testing.B) {
	b.StopTimer()
	InitializeRuntime(b)
	defer CleanupRuntime(b)
	sessionOptions := getTensorRTSessionOptions(b)
	defer sessionOptions.Destroy()
	benchmarkBigSessionWithOptions(b, sessionOptions)
}

func getCoreMLSessionOptions(t testing.TB) *SessionOptions {
	options, e := NewSessionOptions()
	if e != nil {
		t.Logf("Error creating session options: %s\n", e)
		t.FailNow()
	}
	e = options.AppendExecutionProviderCoreML(0)
	if e != nil {
		options.Destroy()
		t.Skipf("Couldn't enable CoreML: %s. This may be due to your system "+
			"or onnxruntime library version not supporting CoreML.\n", e)
	}
	return options
}

func TestCoreMLSession(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)
	sessionOptions := getCoreMLSessionOptions(t)
	defer sessionOptions.Destroy()
	testBigSessionWithOptions(t, sessionOptions)
}

func BenchmarkCoreMLSession(b *testing.B) {
	b.StopTimer()
	InitializeRuntime(b)
	defer CleanupRuntime(b)
	sessionOptions := getCoreMLSessionOptions(b)
	defer sessionOptions.Destroy()
	benchmarkBigSessionWithOptions(b, sessionOptions)
}

func getDirectMLSessionOptions(t testing.TB) *SessionOptions {
	options, e := NewSessionOptions()
	if e != nil {
		t.Logf("Error creating session options: %s\n", e)
		t.FailNow()
	}
	e = options.AppendExecutionProviderDirectML(0)
	if e != nil {
		options.Destroy()
		t.Skipf("Couldn't enable DirectML: %s. This may be due to your "+
			"system or onnxruntime library version not supporting DirectML.\n",
			e)
	}
	return options
}

func TestDirectMLSession(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)
	sessionOptions := getDirectMLSessionOptions(t)
	defer sessionOptions.Destroy()
	testBigSessionWithOptions(t, sessionOptions)
}

func BenchmarkDirectMLSession(b *testing.B) {
	InitializeRuntime(b)
	defer CleanupRuntime(b)
	sessionOptions := getDirectMLSessionOptions(b)
	defer sessionOptions.Destroy()
	benchmarkBigSessionWithOptions(b, sessionOptions)
}
