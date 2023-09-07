package onnxruntime_go

import (
	"encoding/json"
	"fmt"
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
			// Arbitrarily chosen precision.
			if diff >= 0.00000001 {
				return fmt.Errorf("Data element %d doesn't match: %f vs %v",
					i, a[i], b[i])
			}
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

func TestBadTensorShapes(t *testing.T) {
	InitializeRuntime(t)
	defer DestroyEnvironment()
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
