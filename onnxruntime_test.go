package onnxruntime_go

import (
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
	if runtime.GOARCH == "amd64" && runtime.GOOS == "darwin" {
		return "test_data/onnxruntime_amd64.dylib"
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
		t.Fatalf("Failed setting up onnxruntime environment: %s\n", e)
	}
}

// Should be called at the end of each test to de-initialize the runtime.
func CleanupRuntime(t testing.TB) {
	e := DestroyEnvironment()
	if e != nil {
		t.Fatalf("Error cleaning up environment: %s\n", e)
	}
}

// Returns nil if a and b are within a small delta of one another, otherwise
// returns an error indicating their values.
func floatsEqual(a, b float32) error {
	diff := a - b
	if diff < 0 {
		diff = -diff
	}
	// Arbitrarily chosen precision. (Unfortunately, going higher than this may
	// cause test failures, since the Sum operator doesn't have the same
	// results as doing sums purely in Go.)
	if diff >= 0.000001 {
		return fmt.Errorf("Values differ by too much: %f vs %f", a, b)
	}
	return nil
}

// Returns an error if any element between a and b don't match.
func allFloatsEqual(a, b []float32) error {
	if len(a) != len(b) {
		return fmt.Errorf("Length mismatch: %d vs %d", len(a), len(b))
	}
	for i := range a {
		e := floatsEqual(a[i], b[i])
		if e != nil {
			return fmt.Errorf("Data element %d doesn't match: %s", i, e)
		}
	}
	return nil
}

// Returns an empty tensor with the given type and shape, or fails the test on
// error.
func newTestTensor[T TensorData](t testing.TB, s Shape) *Tensor[T] {
	toReturn, e := NewEmptyTensor[T](s)
	if e != nil {
		t.Fatalf("Failed creating empty tensor with shape %s: %s\n", s, e)
	}
	return toReturn
}

func TestGetVersion(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)
	version := GetVersion()
	if version == "" {
		t.Fatalf("Not found version onnxruntime library")
	}
	t.Logf("Found onnxruntime library version: %s\n", version)
}

func TestTensorTypes(t *testing.T) {
	type myFloat float64
	dataType := TensorElementDataType(GetTensorElementDataType[myFloat]())
	expected := TensorElementDataType(TensorElementDataTypeDouble)
	if dataType != expected {
		t.Fatalf("Expected float64 data type to be %d (%s), got %d (%s)\n",
			expected, expected, dataType, dataType)
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
		t.Fatalf("Failed creating %s uint8 tensor: %s\n", s, e)
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
		t.Fatalf("Modifying the original shape incorrectly changed the " +
			"tensor's shape.\n")
	}

	// Try making a tensor with a different data type.
	s = NewShape(2, 5)
	data := []float32{1.0}
	_, e = NewTensor(s, data)
	if e == nil {
		t.Fatalf("Didn't get error when creating a tensor with too little " +
			"data.\n")
	}
	t.Logf("Got expected error when creating a tensor without enough data: "+
		"%s\n", e)

	// It shouldn't be an error to create a tensor with too *much* underlying
	// data; we'll just use the first portion of it.
	data = []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
	tensor2, e := NewTensor(s, data)
	if e != nil {
		t.Fatalf("Error creating tensor with data: %s\n", e)
	}
	defer tensor2.Destroy()
	// Make sure the tensor's internal slice only refers to the part we care
	// about, and not the entire slice.
	if len(tensor2.GetData()) != 10 {
		t.Fatalf("New tensor data contains %d elements, when it should "+
			"contain 10.\n", len(tensor2.GetData()))
	}
}

func TestBadTensorShapes(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)
	s := NewShape()
	_, e := NewEmptyTensor[float64](s)
	if e == nil {
		t.Fatalf("Didn't get an error when creating a tensor with an empty " +
			"shape.\n")
	}
	t.Logf("Got expected error when creating a tensor with an empty shape: "+
		"%s\n", e)
	s = NewShape(10, 0, 10)
	_, e = NewEmptyTensor[uint16](s)
	if e == nil {
		t.Fatalf("Didn't get an error when creating a tensor with a shape " +
			"containing a 0 dimension.\n")
	}
	t.Logf("Got expected error when creating a tensor with a 0 dimension: "+
		"%s\n", e)
	s = NewShape(10, 10, -10)
	_, e = NewEmptyTensor[int32](s)
	if e == nil {
		t.Fatalf("Didn't get an error when creating a tensor with a negative" +
			" dimension.\n")
	}
	t.Logf("Got expected error when creating a tensor with a negative "+
		"dimension: %s\n", e)
	s = NewShape(10, -10, -10)
	_, e = NewEmptyTensor[uint64](s)
	if e == nil {
		t.Fatalf("Didn't get an error when creating a tensor with two " +
			"negative dimensions.\n")
	}
	t.Logf("Got expected error when creating a tensor with two negative "+
		"dimensions: %s\n", e)
	s = NewShape(int64(1)<<62, 1, int64(1)<<62)
	_, e = NewEmptyTensor[float32](s)
	if e == nil {
		t.Fatalf("Didn't get an error when creating a tensor with an " +
			"overflowing shape.\n")
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
		t.Fatalf("Error creating tensor: %s\n", e)
	}
	clone, e := originalTensor.Clone()
	if e != nil {
		t.Fatalf("Error cloning tensor: %s\n", e)
	}
	if !clone.GetShape().Equals(originalTensor.GetShape()) {
		t.Fatalf("Clone shape (%s) doesn't match original shape (%s)\n",
			clone.GetShape(), originalTensor.GetShape())
	}
	cloneData := clone.GetData()
	for i := range originalData {
		if cloneData[i] != originalData[i] {
			t.Fatalf("Clone data incorrect at index %d: %f (expected %f)\n",
				i, cloneData[i], originalData[i])
		}
	}
	cloneData[2] = 1337
	if originalData[2] != 3 {
		t.Fatalf("Modifying clone data effected the original.\n")
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
			t.Fatalf("a[%d] = %f, expected it to be set to 0.\n", i, v)
		}
	}

	// Do the same basic test with a CustomDataTensor
	shape := NewShape(2, 3, 4, 5)
	customData := randomBytes(123, 2*shape.FlattenedSize())
	b, e := NewCustomDataTensor(shape, customData, TensorElementDataTypeUint16)
	if e != nil {
		t.Fatalf("Error creating custom data tensor: %s\n", e)
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
			t.Fatalf("b[%d] = %d, expected it to be set to 0.\n", i, v)
		}
	}
}

// This test makes sure that functions taking .onnx data don't crash when
// passed an empty slice. (This used to be a bug.)
func TestEmptyONNXFiles(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)
	inputNames := []string{"whatever"}
	outputNames := []string{"whatever_out"}
	dummyIn := newTestTensor[float32](t, NewShape(1))
	defer dummyIn.Destroy()
	dummyOut := newTestTensor[float32](t, NewShape(1))
	defer dummyOut.Destroy()
	inputTensors := []Value{dummyIn}
	outputTensors := []Value{dummyOut}
	_, e := NewAdvancedSessionWithONNXData([]byte{}, inputNames, outputNames,
		inputTensors, outputTensors, nil)
	if e == nil {
		// Really we're checking for a panic due to the empty slice, rather
		// than a nil error.
		t.Fatalf("Didn't get expected error when creating session.\n")
	}
	t.Logf("Got expected error creating session with no ONNX content: %s\n", e)
	_, e = NewDynamicAdvancedSessionWithONNXData([]byte{}, inputNames,
		outputNames, nil)
	if e == nil {
		t.Fatalf("Didn't get expected error when creating dynamic advanced " +
			"session.\n")
	}
	t.Logf("Got expected error when creating dynamic session with no ONNX "+
		"content: %s\n", e)
	_, _, e = GetInputOutputInfoWithONNXData([]byte{})
	if e == nil {
		t.Fatalf("Didn't get expected error when getting input/output info " +
			"with no ONNX content.\n")
	}
	t.Logf("Got expected error when getting input/output info with no "+
		"ONNX content: %s\n", e)
	_, e = GetModelMetadataWithONNXData([]byte{})
	if e == nil {
		t.Fatalf("Didn't get expected error when getting metadata with no " +
			"ONNX content.\n")
	}
	t.Logf("Got expected error when getting metadata with no ONNX "+
		"content: %s\n", e)
}

func TestLegacyAPI(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)

	// We'll use this network simply due to its simple input and output format,
	// as well as it using the same data type for inputs and outputs. See
	// TestNonAsciiPath for more comments.
	filePath := "test_data/example ż 大 김.onnx"
	inputData := []int32{12, 21}
	input, e := NewTensor(NewShape(1, 2), inputData)
	if e != nil {
		t.Fatalf("Error creating input tensor: %s\n", e)
	}
	defer input.Destroy()
	output := newTestTensor[int32](t, NewShape(1))
	defer output.Destroy()

	session, e := NewSession[int32](filePath, []string{"in"}, []string{"out"},
		[]*Tensor[int32]{input}, []*Tensor[int32]{output})
	if e != nil {
		t.Fatalf("Error creating sesion via legacy API: %s\n", e)
	}
	e = session.Run()
	if e != nil {
		t.Fatalf("Error running session: %s\n", e)
	}
	expected := inputData[0] + inputData[1]
	result := output.GetData()[0]
	if result != expected {
		t.Errorf("Incorrect result. Expected %d, got %d.\n", expected, result)
	}
}

func TestLegacyAPIDynamic(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)
	filePath := "test_data/example ż 大 김.onnx"
	inputData := []int32{12, 21}
	input, e := NewTensor(NewShape(1, 2), inputData)
	if e != nil {
		t.Fatalf("Error creating input tensor: %s\n", e)
	}
	defer input.Destroy()
	output := newTestTensor[int32](t, NewShape(1))
	defer output.Destroy()

	session, e := NewDynamicSession[int32, int32](filePath,
		[]string{"in"}, []string{"out"})
	if e != nil {
		t.Fatalf("Error creating sesion via legacy API: %s\n", e)
	}
	e = session.Run([]*Tensor[int32]{input}, []*Tensor[int32]{output})
	if e != nil {
		t.Fatalf("Error running session: %s\n", e)
	}
	expected := inputData[0] + inputData[1]
	result := output.GetData()[0]
	if result != expected {
		t.Errorf("Incorrect result. Expected %d, got %d.\n", expected, result)
	}
}

func TestEnableDisableTelemetry(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)

	e := EnableTelemetry()
	if e != nil {
		t.Errorf("Error enabling onnxruntime telemetry: %s\n", e)
	}
	e = DisableTelemetry()
	if e != nil {
		t.Errorf("Error disabling onnxruntime telemetry: %s\n", e)
	}
	e = EnableTelemetry()
	if e != nil {
		t.Errorf("Error re-enabling onnxruntime telemetry after "+
			"disabling: %s\n", e)
	}
}

func TestArbitraryTensors(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)

	tensorShape := NewShape(2, 2)
	tensorA, e := NewTensor(tensorShape, []uint8{1, 2, 3, 4})
	if e != nil {
		t.Fatalf("Error creating uint8 tensor: %s\n", e)
	}
	defer tensorA.Destroy()
	tensorB, e := NewTensor(tensorShape, []float64{5, 6, 7, 8})
	if e != nil {
		t.Fatalf("Error creating float64 tensor: %s\n", e)
	}
	defer tensorB.Destroy()
	tensorC, e := NewTensor(tensorShape, []int16{9, 10, 11, 12})
	if e != nil {
		t.Fatalf("Error creating int16 tensor: %s\n", e)
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
		t.Fatalf("Failed creating input B: %s\n", e)
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
		t.Fatalf("Expected tensor to contain %d elements, got %d elements.\n",
			len(expectedContent), len(data))
	}
	for i, v := range expectedContent {
		if v != data[i] {
			t.Fatalf("Data mismatch at index %d: expected %v, got %v\n", i, v,
				data[i])
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

	// Decided to toss in an "ArbitraryTensor" here to ensure that it remains
	// compatible with Value in the future.
	session, e := NewAdvancedSession("test_data/example_multitype.onnx",
		[]string{"InputA", "InputB"}, []string{"OutputA", "OutputB"},
		[]Value{inputA, inputB}, []ArbitraryTensor{outputA, outputB}, nil)
	if e != nil {
		t.Fatalf("Failed creating session: %s\n", e)
	}
	defer session.Destroy()
	e = session.Run()
	if e != nil {
		t.Fatalf("Error running session: %s\n", e)
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
		e = session.Run([]Value{aInputs[i], bInputs[i]},
			[]Value{aOutputs[i], bOutputs[i]})
		if e != nil {
			t.Fatalf("Failed running session for test %d: %s\n", i, e)
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

	session, e := NewDynamicAdvancedSession("test_data/example_multitype.onnx",
		[]string{"InputA", "InputB"}, []string{"OutputA", "OutputB"}, nil)
	if e != nil {
		t.Fatalf("Error creating session: %s\n", e)
	}
	defer session.Destroy()

	// Actually create the inputs and run the tests.
	aInput, bInput := randomMultitypeInputs(t, 999)
	var outputs [2]Value
	e = session.Run([]Value{aInput, bInput}, outputs[:])
	if e != nil {
		t.Fatalf("Failed running session: %s\n", e)
	}
	defer func() {
		for _, output := range outputs {
			output.Destroy()
		}
	}()

	expectedA, expectedB := getExpectedMultitypeOutputs(aInput, bInput)
	expectedShape := NewShape(1, 2, 2)
	outputA, ok := outputs[0].(*Tensor[int16])
	if !ok {
		t.Fatalf("Expected outputA to be of type %T, got of type %T\n",
			outputA, outputs[0])
	}
	if !outputA.shape.Equals(expectedShape) {
		t.Fatalf("Expected outputA to be of shape %s, got of shape %s\n",
			expectedShape, outputA.shape)
	}
	verifyTensorData(t, outputA, expectedA)

	outputB, ok := outputs[1].(*Tensor[int64])
	expectedShape = NewShape(1, 1, 1)
	if !ok {
		t.Fatalf("Expected outputB to be of type %T, got of type %T\n",
			outputB, outputs[1])
	}
	if !outputB.shape.Equals(expectedShape) {
		t.Fatalf("Expected outputB to be of shape %s, got of shape %s\n",
			expectedShape, outputB.shape)
	}
	verifyTensorData(t, outputB, expectedB)
}

// Makes sure that the sum of each vector in the input tensor matches the
// corresponding scalar in the output tensor. Used when testing tensors with
// unknown batch dimensions.
// NOTE: Destroys the input and output tensors before returning, regardless of
// test success.
func checkVectorSum(input *Tensor[float32], output *Tensor[float32],
	t testing.TB) {
	defer input.Destroy()
	defer output.Destroy()
	// Make sure the sizes are what we expect.
	inputShape := input.GetShape()
	outputShape := output.GetShape()
	if len(inputShape) != 2 {
		t.Fatalf("Expected a 2-dimensional input shape, got %v\n", inputShape)
	}
	if len(outputShape) != 1 {
		t.Fatalf("Expected 1-dimensional output shape, got %v\n", outputShape)
	}
	if inputShape[0] != outputShape[0] {
		t.Fatalf("Input and output batch dimensions don't match (%d vs %d)\n",
			inputShape[0], outputShape[0])
	}

	// Compute the sums in Go
	batchSize := inputShape[0]
	vectorLength := inputShape[1]
	expectedSums := make([]float32, batchSize)
	for i := int64(0); i < batchSize; i++ {
		inputVector := input.GetData()[i*vectorLength : (i+1)*vectorLength]
		sum := float32(0.0)
		for _, v := range inputVector {
			sum += v
		}
		expectedSums[i] = sum
	}

	e := allFloatsEqual(expectedSums, output.GetData())
	if e != nil {
		t.Fatalf("ONNX-produced sums don't match CPU-produced sums: %s\n", e)
	}
}

func TestDynamicInputOutputAxes(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)

	netPath := "test_data/example_dynamic_axes.onnx"
	session, e := NewDynamicAdvancedSession(netPath,
		[]string{"input_vectors"}, []string{"output_scalars"}, nil)
	if e != nil {
		t.Fatalf("Error loading %s: %s\n", netPath, e)
	}
	defer session.Destroy()
	maxBatchSize := 99
	// The example network takes a dynamic batch size of vectors containing 10
	// elements each.
	dataBuffer := make([]float32, maxBatchSize*10)

	// Try running the session with many different batch sizes
	for i := 11; i <= maxBatchSize; i += 11 {
		// Create an input with the new batch size.
		inputShape := NewShape(int64(i), 10)
		input, e := NewTensor(inputShape, dataBuffer)
		if e != nil {
			t.Fatalf("Error creating input tensor with shape %v: %s\n",
				inputShape, e)
		}

		// Populate the input with new random floats.
		fillRandomFloats(input.GetData(), 1234)

		// Run the session; make onnxruntime allocate the output tensor for us.
		outputs := []Value{nil}
		e = session.Run([]Value{input}, outputs)
		if e != nil {
			input.Destroy()
			t.Fatalf("Error running the session with batch size %d: %s\n",
				i, e)
		}

		// The checkVectorSum function will destroy the input and output tensor
		// regardless of their correctness.
		checkVectorSum(input, outputs[0].(*Tensor[float32]), t)
		input.Destroy()
		t.Logf("Batch size %d seems OK!\n", i)
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
	e = session.Run([]Value{inputA, inputB}, []Value{wrongTypeTensor, outputB})
	if e == nil {
		t.Fatalf("Didn't get expected error when passing a float32 tensor in" +
			" place of an int16 output tensor.\n")
	}
	t.Logf("Got expected error when passing a float32 tensor in place of an "+
		"int16 output tensor: %s\n", e)
	e = session.Run([]Value{inputA, wrongTypeTensor},
		[]Value{outputA, outputB})
	if e == nil {
		t.Fatalf("Didn't get expected error when passing a float32 tensor in" +
			" place of a float64 input tensor.\n")
	}
	t.Logf("Got expected error when passing a float32 tensor in place of a "+
		"float64 input tensor: %s\n", e)

	// Make sure that passing a tensor with the wrong shape but correct type
	// will cause an error rather than a crash, when using as an input or an
	// output.
	wrongShapeInput := newTestTensor[uint8](t, NewShape(22))
	defer wrongShapeInput.Destroy()
	e = session.Run([]Value{wrongShapeInput, inputB},
		[]Value{outputA, outputB})
	if e == nil {
		t.Fatalf("Didn't get expected error when running with an incorrectly" +
			" shaped input.\n")
	}
	t.Logf("Got expected error when running with an incorrectly shaped "+
		"input: %s\n", e)
	wrongShapeOutput := newTestTensor[int64](t, NewShape(1, 1, 1, 1, 1, 1))
	defer wrongShapeOutput.Destroy()
	e = session.Run([]Value{inputA, inputB},
		[]Value{outputA, wrongShapeOutput})
	if e == nil {
		t.Fatalf("Didn't get expected error when running with an incorrectly" +
			" shaped output.\n")
	}
	t.Logf("Got expected error when running with an incorrectly shaped "+
		"output: %s\n", e)

	e = session.Run([]Value{inputA, inputB}, []Value{outputA, outputB})
	if e != nil {
		t.Fatalf("Got error attempting to (correctly) Run a session after "+
			"attempting to use incorrect inputs or outputs: %s\n", e)
	}
}

func TestGetInputOutputInfo(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)
	file := "test_data/example_several_inputs_and_outputs.onnx"
	inputs, outputs, e := GetInputOutputInfo(file)
	if e != nil {
		t.Fatalf("Error getting input and output info for %s: %s\n", file, e)
	}
	if len(inputs) != 3 {
		t.Fatalf("Expected 3 inputs, got %d\n", len(inputs))
	}
	if len(outputs) != 2 {
		t.Fatalf("Expected 2 outputs, got %d\n", len(outputs))
	}
	for i, v := range inputs {
		t.Logf("Input %d: %s\n", i, &v)
	}
	for i, v := range outputs {
		t.Logf("Output %d: %s\n", i, &v)
	}

	if outputs[1].Name != "output 2" {
		t.Errorf("Incorrect output 1 name: %s, expected \"output 2\"\n",
			outputs[1].Name)
	}
	expectedShape := NewShape(1, 2, 3, 4, 5)
	if !outputs[1].Dimensions.Equals(expectedShape) {
		t.Errorf("Incorrect output 1 shape: %s, expected %s\n",
			outputs[1].Dimensions, expectedShape)
	}
	var expectedType TensorElementDataType = TensorElementDataTypeDouble
	if outputs[1].DataType != expectedType {
		t.Errorf("Incorrect output 1 data type: %s, expected %s\n",
			outputs[1].DataType, expectedType)
	}
	if inputs[0].Name != "input 1" {
		t.Errorf("Incorrect input 0 name: %s, expected \"input 1\"\n",
			inputs[0].Name)
	}
	expectedShape = NewShape(2, 5, 2, 5)
	if !inputs[0].Dimensions.Equals(expectedShape) {
		t.Errorf("Incorrect input 0 shape: %s, expected %s\n",
			inputs[0].Dimensions, expectedShape)
	}
	expectedType = TensorElementDataTypeInt32
	if inputs[0].DataType != expectedType {
		t.Errorf("Incorrect input 0 data type: %s, expected %s\n",
			inputs[0].DataType, expectedType)
	}
}

func TestModelMetadata(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)
	file := "test_data/example_big_compute.onnx"
	metadata, e := GetModelMetadata(file)
	if e != nil {
		t.Fatalf("Error getting metadata for %s: %s\n", file, e)
	}
	// We'll just test Destroy once; after this we won't check its return value
	e = metadata.Destroy()
	if e != nil {
		t.Fatalf("Error destroying metadata: %s\n", e)
	}

	// Try getting the metadata from a session instead of from a file.
	// NOTE: All of the expected values here were manually set using the
	// test_data/modify_metadata.py script after generating the network. See
	// that script for the expected values of each of the metadata accesors.
	session, e := NewDynamicAdvancedSession(file, []string{"Input"},
		[]string{"Output"}, nil)
	if e != nil {
		t.Fatalf("Error creating session: %s\n", e)
	}
	defer session.Destroy()
	metadata, e = session.GetModelMetadata()
	if e != nil {
		t.Fatalf("Error getting metadata from DynamicAdvancedSession: %s\n", e)
	}
	defer metadata.Destroy()
	producerName, e := metadata.GetProducerName()
	if e != nil {
		t.Errorf("Error getting producer name: %s\n", e)
	} else {
		t.Logf("Got producer name: %s\n", producerName)
	}
	graphName, e := metadata.GetGraphName()
	if e != nil {
		t.Errorf("Error getting graph name: %s\n", e)
	} else {
		t.Logf("Got graph name: %s\n", graphName)
	}
	domainStr, e := metadata.GetDomain()
	if e != nil {
		t.Errorf("Error getting domain: %s\n", e)
	} else {
		t.Logf("Got domain: %s\n", domainStr)
		if domainStr != "test domain" {
			t.Errorf("Incorrect domain string, expected \"test domain\"\n")
		}
	}
	description, e := metadata.GetDescription()
	if e != nil {
		t.Errorf("Error getting description: %s\n", e)
	} else {
		t.Logf("Got description: %s\n", description)
	}
	version, e := metadata.GetVersion()
	if e != nil {
		t.Errorf("Error getting version: %s\n", e)
	} else {
		t.Logf("Got version: %d\n", version)
		if version != 1337 {
			t.Errorf("Incorrect version number, expected 1337\n")
		}
	}
	mapKeys, e := metadata.GetCustomMetadataMapKeys()
	if e != nil {
		t.Fatalf("Error getting custom metadata keys: %s\n", e)
	}
	t.Logf("Got %d custom metadata map keys.\n", len(mapKeys))
	if len(mapKeys) != 2 {
		t.Errorf("Incorrect number of custom metadata keys, expected 2")
	}
	for _, k := range mapKeys {
		value, present, e := metadata.LookupCustomMetadataMap(k)
		if e != nil {
			t.Errorf("Error looking up key %s in custom metadata: %s\n", k, e)
		} else {
			if !present {
				t.Errorf("LookupCustomMetadataMap didn't return true for a " +
					"key that should be present in the map\n")
			}
			t.Logf("  Metadata key \"%s\" = \"%s\"\n", k, value)
		}
	}
	badValue, present, e := metadata.LookupCustomMetadataMap("invalid key")
	if len(badValue) != 0 {
		t.Fatalf("Didn't get an empty string when looking up an invalid "+
			"metadata key, got \"%s\" instead\n", badValue)
	}
	if present {
		t.Errorf("LookupCustomMetadataMap didn't return false for a key that" +
			" isn't in the map\n")
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

func fillRandomFloats(dst []float32, seed int64) {
	rng := rand.New(rand.NewSource(seed))
	for i := range dst {
		dst[i] = rng.Float32()
	}
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
		t.Fatalf("Error creating uint16 CustomDataTensor: %s\n", e)
	}
	shape[0] = 6
	if v.GetShape().Equals(shape) {
		t.Fatalf("CustomDataTensor didn't properly clone its shape")
	}
	e = v.Destroy()
	if e != nil {
		t.Fatalf("Error destroying CustomDataTensor: %s\n", e)
	}
	tensorData = randomBytes(1234, 2*shape.FlattenedSize())
	v, e = NewCustomDataTensor(shape, tensorData, TensorElementDataTypeFloat16)
	if e != nil {
		t.Fatalf("Error creating float16 tensor: %s\n", e)
	}
	e = v.Destroy()
	if e != nil {
		t.Fatalf("Error destroying float16 tensor: %s\n", e)
	}
	// Make sure we don't fail if providing more data than necessary
	shape[0] = 1
	v, e = NewCustomDataTensor(shape, tensorData,
		TensorElementDataTypeBFloat16)
	if e != nil {
		t.Fatalf("Got error when creating a tensor with more data than "+
			"necessary: %s\n", e)
	}
	v.Destroy()

	// Make sure we fail when using a bad shape
	shape = NewShape(0, -1, -2)
	v, e = NewCustomDataTensor(shape, tensorData, TensorElementDataTypeFloat16)
	if e == nil {
		v.Destroy()
		t.Fatalf("Didn't get error when creating custom tensor with an " +
			"invalid shape\n")
	}
	t.Logf("Got expected error creating tensor with invalid shape: %s\n", e)
	shape = NewShape(1, 2, 3, 4, 5)
	tensorData = []byte{1, 2, 3, 4}
	v, e = NewCustomDataTensor(shape, tensorData, TensorElementDataTypeUint8)
	if e == nil {
		v.Destroy()
		t.Fatalf("Didn't get error when creating custom tensor with too " +
			"little data\n")
	}
	t.Logf("Got expected error when creating custom data tensor with "+
		"too little data: %s\n", e)

	// Make sure we fail when using a bad type
	tensorData = []byte{1, 2, 3, 4, 5, 6, 7, 8}
	badType := TensorElementDataType(0xffffff)
	v, e = NewCustomDataTensor(NewShape(2), tensorData, badType)
	if e == nil {
		v.Destroy()
		t.Fatalf("Didn't get error when creating tensor with bad type\n")
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
		t.Fatalf("Error creating input tensor: %s\n", e)
	}
	defer inputTensor.Destroy()
	outputTensor, e := NewCustomDataTensor(NewShape(1, 2, 2, 2), outputData,
		TensorElementDataTypeBFloat16)
	if e != nil {
		t.Fatalf("Error creating output tensor: %s\n", e)
	}
	defer outputTensor.Destroy()

	session, e := NewAdvancedSession("test_data/example_float16.onnx",
		[]string{"InputA"}, []string{"OutputA"},
		[]Value{inputTensor}, []Value{outputTensor}, nil)
	if e != nil {
		t.Fatalf("Error creating session: %s\n", e)
	}
	defer session.Destroy()
	e = session.Run()
	if e != nil {
		t.Fatalf("Error running session: %s\n", e)
	}
	for i := range outputData {
		if outputData[i] != expectedOutput[i] {
			t.Fatalf("Incorrect output byte at index %d: 0x%02x (expected "+
				"0x%02x)\n", i, outputData[i], expectedOutput[i])
		}
	}
}

// Returns a 10-element tensor randomly filled values using the given rng seed.
func randomSmallTensor(seed int64, t testing.TB) *Tensor[float32] {
	toReturn, e := NewEmptyTensor[float32](NewShape(10))
	if e != nil {
		t.Fatalf("Error creating small tensor: %s\n", e)
	}
	fillRandomFloats(toReturn.GetData(), seed)
	return toReturn
}

func TestONNXSequence(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)
	sequenceLength := int64(123)

	values := make([]Value, sequenceLength)
	for i := range values {
		values[i] = randomSmallTensor(int64(i)+123, t)
	}
	defer func() {
		for _, v := range values {
			v.Destroy()
		}
	}()
	sequence, e := NewSequence(values)
	if e != nil {
		t.Fatalf("Error creating sequence: %s\n", e)
	}
	defer sequence.Destroy()
	sequenceContents, e := sequence.GetValues()
	if e != nil {
		t.Fatalf("Error getting sequence contents: %s\n", e)
	}
	if int64(len(sequenceContents)) != sequenceLength {
		t.Fatalf("Got %d values in sequence, expected %d\n",
			len(sequenceContents), sequenceLength)
	}
	if sequence.GetONNXType() != ONNXTypeSequence {
		t.Fatalf("Got incorrect ONNX type for sequence: %s\n",
			sequence.GetONNXType())
	}
	// Make sure we adhere to what I wrote in the docs
	if !sequence.GetShape().Equals(NewShape(sequenceLength)) {
		t.Fatalf("Sequence.GetShape() returned incorrect shape: %s\n",
			sequence.GetShape())
	}

	selectedIndex := 44
	selectedValue := sequenceContents[selectedIndex]
	if selectedValue.GetONNXType() != ONNXTypeTensor {
		t.Fatalf("Got incorrect ONNXType for value at index %d: "+
			"expected %s, got %s\n", selectedIndex, ONNXType(ONNXTypeTensor),
			selectedValue.GetONNXType())
	}
}

func TestBadSequences(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)

	// Sequences containing no elements or nil entries shouldn't be allowed
	_, e := NewSequence([]Value{})
	if e == nil {
		t.Fatalf("Didn't get expected error when creating an empty sequence\n")
	}
	t.Logf("Got expected error when creating an empty sequence: %s\n", e)
	_, e = NewSequence([]Value{nil})
	if e == nil {
		t.Fatalf("Didn't get expected error when creating sequence with a " +
			"nil entry.\n")
	}
	t.Logf("Got expected error when creating sequence with nil entry: %s\n", e)

	// Sequences containing mixed data types shouldn't be allowed
	tensor := randomSmallTensor(1337, t)
	defer tensor.Destroy()
	innerSequence, e := NewSequence([]Value{tensor})
	if e != nil {
		t.Fatalf("Error creating 1-element sequence: %s\n", e)
	}
	defer innerSequence.Destroy()
	_, e = NewSequence([]Value{tensor, innerSequence})
	if e == nil {
		t.Fatalf("Didn't get expected error when attempting to create a "+
			"mixed sequence: %s\n", e)
	}
	t.Logf("Got expected error when attempting a mixed sequence: %s\n", e)

	// Nested sequences also aren't allowed; the C API docs don't seem to
	// mention this either.
	_, e = NewSequence([]Value{innerSequence, innerSequence})
	if e == nil {
		t.Fatalf("Didn't get an error creating a sequence with nested " +
			"sequences.\n")
	}
	t.Logf("Got expected error when creating a sequence with nested "+
		"sequences: %s\n", e)
}

func TestMap(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)

	testGoMap := map[int64]float64{
		123: 456.7,
		789: 123.4,
	}
	m, e := NewMapFromGoMap(testGoMap)
	if e != nil {
		t.Fatalf("Error creating onnx map from Go map: %s\n", e)
	}
	defer m.Destroy()
	keys, values, e := m.GetKeysAndValues()
	if e != nil {
		t.Fatalf("Error getting map keys and values: %s\n", e)
	}

	// In real code I almost certainly would do these type assertions without
	// the checks, and just panic if it was wrong. But it makes sense in a test
	keysTensor, ok := keys.(*Tensor[int64])
	if !ok {
		t.Fatalf("Keys weren't a uint32 tensor, but %s\n",
			TensorElementDataType(keysTensor.DataType()))
	}
	valuesTensor, ok := values.(*Tensor[float64])
	if !ok {
		t.Fatalf("Values weren't a float64 tensor, but %s\n",
			TensorElementDataType(valuesTensor.DataType()))
	}

	if !keysTensor.GetShape().Equals(valuesTensor.GetShape()) {
		t.Fatalf("Key and value tensor shapes don't match: %s vs %s\n",
			keysTensor.GetShape(), valuesTensor.GetShape())
	}

	for i, k := range keysTensor.GetData() {
		v := valuesTensor.GetData()[i]
		e = floatsEqual(float32(v), float32(testGoMap[k]))
		if e != nil {
			t.Errorf("Value for key %d doesn't match: %s\n", k, e)
		}
	}
}

func TestBadMaps(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)

	// There are many, many ways I've found to create a bad map. This test only
	// checks a few of them.

	// We should get an error for an empty map, right? (I don't think the docs
	// specify at the moment.)
	_, e := NewMapFromGoMap(map[int64]float32{})
	if e == nil {
		t.Fatalf("Didn't get expected error creating empty map.\n")
	}
	t.Logf("Got expected error when creating empty map: %s\n", e)

	// Floats aren't supported as keys.
	floatKeysTensor := newTestTensor[float32](t, NewShape(10))
	defer floatKeysTensor.Destroy()
	floatValuesTensor := newTestTensor[float32](t, NewShape(10))
	defer floatValuesTensor.Destroy()
	_, e = NewMap(floatKeysTensor, floatValuesTensor)
	if e == nil {
		t.Fatalf("Didn't get expected error when using float map keys.\n")
	}
	t.Logf("Got expected error when using float map keys: %s\n", e)

	// The length of keys and values must match.
	tooManyKeysTensor := newTestTensor[int64](t, NewShape(16))
	for i := range tooManyKeysTensor.GetData() {
		tooManyKeysTensor.GetData()[i] = int64(i)
	}
	defer tooManyKeysTensor.Destroy()
	_, e = NewMap(tooManyKeysTensor, floatValuesTensor)
	if e == nil {
		t.Fatalf("Didn't get expected error when map keys and values are " +
			"different sizes.\n")
	}
	t.Logf("Got expected error when keys and values lengths mismatch: %s\n", e)
}

func TestSklearnNetwork(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)

	// These inputs and outputs were taken from the information printed by
	// test_data/generate_sklearn_network.py
	inputShape := NewShape(6, 4)
	inputValues := []float32{
		5.9, 3.0, 5.1, 1.8,
		6.8, 2.8, 4.8, 1.4,
		6.3, 2.3, 4.4, 1.3,
		6.5, 3.0, 5.5, 1.8,
		7.7, 2.8, 6.7, 2.0,
		5.5, 2.5, 4.0, 1.3,
	}

	// "output_label": A tensor of an int64 label per set of 4 inputs
	expectedPredictions := []int64{2, 1, 1, 2, 2, 1}

	// "output_probability": A sequence of maps, mapping each int64 label to a
	// float64 output. We'll just store them in order here.
	outputProbabilities := []map[int64]float32{
		{0: 0.0, 1: 0.12999998033046722, 2: 0.8699994683265686},
		{0: 0.0, 1: 0.7699995636940002, 2: 0.23000003397464752},
		{0: 0.0, 1: 0.969999372959137, 2: 0.029999999329447746},
		{0: 0.0, 1: 0.0, 2: 0.9999993443489075},
		{0: 0.0, 1: 0.0, 2: 0.9999993443489075},
		{0: 0.0, 1: 0.9999993443489075, 2: 0.0},
	}

	modelPath := "test_data/sklearn_randomforest.onnx"
	session, e := NewDynamicAdvancedSession(modelPath, []string{"X"},
		[]string{"output_label", "output_probability"}, nil)
	if e != nil {
		t.Fatalf("Error loading %s: %s\n", modelPath, e)
	}
	defer session.Destroy()

	// The point of this test is to make sure we get the correct types and
	// results when the network allocates the output values.
	outputs := []Value{nil, nil}
	inputTensor, e := NewTensor(inputShape, inputValues)
	if e != nil {
		t.Fatalf("Error creating input tensor: %s\n", e)
	}
	defer inputTensor.Destroy()
	e = session.Run([]Value{inputTensor}, outputs)
	if e != nil {
		t.Fatalf("Error running %s: %s\n", modelPath, e)
	}
	defer func() {
		for _, v := range outputs {
			v.Destroy()
		}
	}()

	// First, check the easy part: the int64 output tensor
	tensorDataType := TensorElementDataType(outputs[0].DataType())
	if tensorDataType != TensorElementDataTypeInt64 {
		t.Fatalf("Expected int64 output tensor, got %s\n", tensorDataType)
	}
	predictionTensor := outputs[0].(*Tensor[int64])
	predictions := predictionTensor.GetData()
	if len(predictions) != len(expectedPredictions) {
		t.Fatalf("Expected %d predictions, got %d\n", len(expectedPredictions),
			len(predictions))
	}
	for i, v := range expectedPredictions {
		actualPrediction := predictions[i]
		if v != actualPrediction {
			t.Errorf("Incorrect prediction at index %d: %d (expected %d)\n",
				i, actualPrediction, v)
		}
	}

	// Next, check the sequence of maps. There is one map giving the fine-
	// grained probabilities for each label. (Predictions is just the entry
	// of each map with the highest probability.)
	sequence, ok := outputs[1].(*Sequence)
	if !ok {
		t.Fatalf("Expected a sequence for the probabilities output, got %s\n",
			outputs[1].GetONNXType())
	}
	probabilityMaps, e := sequence.GetValues()
	if e != nil {
		t.Fatalf("Error getting contents of sequence of maps: %s\n", e)
	}
	if len(probabilityMaps) != len(expectedPredictions) {
		t.Fatalf("Expected a %d-element sequence, got %d\n",
			len(expectedPredictions), len(probabilityMaps))
	}
	for i := range probabilityMaps {
		m, isMap := probabilityMaps[i].(*Map)
		if !isMap {
			t.Fatalf("Output sequence index %d wasn't a map, but a %s\n", i,
				probabilityMaps[i].GetONNXType())
		}
		keys, values, e := m.GetKeysAndValues()
		if e != nil {
			t.Fatalf("Error getting keys and values for map at index %d: %s\n",
				i, e)
		}
		if !keys.GetShape().Equals(values.GetShape()) {
			t.Fatalf("Key and value tensors don't match in shape: %s vs %s\n",
				keys.GetShape(), values.GetShape())
		}
		keysTensor, ok := keys.(*Tensor[int64])
		if !ok {
			t.Fatalf("Keys were not an int64 tensor\n")
		}
		valuesTensor, ok := values.(*Tensor[float32])
		if !ok {
			t.Fatalf("Values were not a float32 tensor\n")
		}
		expectedProbabilities := outputProbabilities[i]
		for j, key := range keysTensor.GetData() {
			v := valuesTensor.GetData()[j]
			e = floatsEqual(expectedProbabilities[key], v)
			if e != nil {
				t.Errorf("Expected values don't match for key %d in map "+
					"index %d: %s\n", key, i, e)
			}
		}
	}
}

// This tests that we're able to read a file containing multi-byte characters
// in the path.
func TestNonAsciiPath(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)

	// The test network just adds two integers and returns the result.
	inputData := []int32{12, 21}
	input, e := NewTensor(NewShape(1, 2), inputData)
	if e != nil {
		t.Fatalf("Error creating input tensor: %s\n", e)
	}
	defer input.Destroy()
	output := newTestTensor[int32](t, NewShape(1))
	defer output.Destroy()

	filePath := "test_data/example ż 大 김.onnx"
	session, e := NewAdvancedSession(filePath, []string{"in"}, []string{"out"},
		[]Value{input}, []Value{output}, nil)
	if e != nil {
		t.Fatalf("Failed creating session for %s: %s\n", filePath, e)
	}

	e = session.Run()
	if e != nil {
		t.Fatalf("Error running %s: %s\n", filePath, e)
	}
	expected := inputData[0] + inputData[1]
	result := output.GetData()[0]
	if result != expected {
		t.Errorf("Running %s gave the wrong result. Expected %d, got %d.\n",
			filePath, expected, result)
	}
}

// This tests that the *WithONNXData method works for loading a session.
// Hopefully this covers most other *WithONNXData variants, since all use the
// same code internally when creating an OrtSession in C.
func TestSessionFromDataBuffer(t *testing.T) {
	// This test is almost a copy of TestNonAsciiPath, since it was fairly
	// simple.
	InitializeRuntime(t)
	defer CleanupRuntime(t)

	inputData := []int32{12, 21}
	input, e := NewTensor(NewShape(1, 2), inputData)
	if e != nil {
		t.Fatalf("Error creating input tensor: %s\n", e)
	}
	defer input.Destroy()
	output := newTestTensor[int32](t, NewShape(1))
	defer output.Destroy()

	filePath := "test_data/example ż 大 김.onnx"
	fileData, e := os.ReadFile(filePath)
	if e != nil {
		t.Fatalf("Error buffering content of %s: %s\n", filePath, e)
	}

	session, e := NewAdvancedSessionWithONNXData(fileData, []string{"in"},
		[]string{"out"}, []Value{input}, []Value{output}, nil)
	if e != nil {
		t.Fatalf("Failed creating session: %s\n", e)
	}
	e = session.Run()
	if e != nil {
		t.Fatalf("Error running session: %s\n", e)
	}
	expected := inputData[0] + inputData[1]
	result := output.GetData()[0]
	if result != expected {
		t.Errorf("Incorrect result. Expected %d, got %d.\n", expected, result)
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
		t.Fatalf("Error creating input tensor: %s\n", e)
	}
	output, e := NewEmptyTensor[float32](NewShape(1, vectorLength))
	if e != nil {
		input.Destroy()
		t.Fatalf("Error creating output tensor: %s\n", e)
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
		[]string{"Input"}, []string{"Output"},
		[]Value{input}, []Value{output}, options)
	if e != nil {
		t.Fatalf("Error creating session: %s\n", e)
	}
	defer session.Destroy()
	e = session.Run()
	if e != nil {
		t.Fatalf("Error running the session: %s\n", e)
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
		[]string{"Input"}, []string{"Output"},
		[]Value{input}, []Value{output}, options)
	if e != nil {
		b.Fatalf("Error creating session: %s\n", e)
	}
	defer session.Destroy()
	b.StartTimer()
	for n := 0; n < b.N; n++ {
		e = session.Run()
		if e != nil {
			b.Fatalf("Error running iteration %d/%d: %s\n", n+1, b.N, e)
		}
	}
}

func TestSessionOptions(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)
	options, e := NewSessionOptions()
	if e != nil {
		t.Fatalf("Error creating session options: %s\n", e)
	}
	defer options.Destroy()
	e = options.SetIntraOpNumThreads(3)
	if e != nil {
		t.Fatalf("Error setting intra-op num threads: %s\n", e)
	}
	e = options.SetInterOpNumThreads(1)
	if e != nil {
		t.Fatalf("Error setting inter-op num threads: %s\n", e)
	}
	e = options.SetCpuMemArena(true)
	if e != nil {
		t.Fatalf("Error setting CPU memory arena: %s\n", e)
	}
	e = options.SetMemPattern(true)
	if e != nil {
		t.Fatalf("Error setting memory pattern: %s\n", e)
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
		b.Fatalf("Error creating options: %s\n", e)
	}
	defer options.Destroy()
	e = options.SetIntraOpNumThreads(nThreads)
	if e != nil {
		b.Fatalf("Error setting intra-op threads to %d: %s\n", nThreads, e)
	}
	e = options.SetInterOpNumThreads(nThreads)
	if e != nil {
		b.Fatalf("Error setting inter-op threads to %d: %s\n", nThreads, e)
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
		t.Fatalf("Error creating SessionOptions: %s\n", e)
	}
	e = sessionOptions.AppendExecutionProviderCUDA(cudaOptions)
	if e != nil {
		sessionOptions.Destroy()
		t.Fatalf("Error setting CUDA execution provider options: %s\n", e)
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
		t.Fatalf("Error creating SessionOptions: %s\n", e)
	}
	e = sessionOptions.AppendExecutionProviderTensorRT(trtOptions)
	if e != nil {
		sessionOptions.Destroy()
		t.Fatalf("Error setting TensorRT execution provider: %s\n", e)
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
		t.Fatalf("Error creating session options: %s\n", e)
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
		t.Fatalf("Error creating session options: %s\n", e)
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
	b.StopTimer()
	InitializeRuntime(b)
	defer CleanupRuntime(b)
	sessionOptions := getDirectMLSessionOptions(b)
	defer sessionOptions.Destroy()
	benchmarkBigSessionWithOptions(b, sessionOptions)
}

func getOpenVINOSessionOptions(t testing.TB) *SessionOptions {
	options, e := NewSessionOptions()
	if e != nil {
		t.Fatalf("Error creating session options: %s\n", e)
	}
	e = options.AppendExecutionProviderOpenVINO(map[string]string{})
	if e != nil {
		options.Destroy()
		t.Skipf("Couldn't enable OpenVINO: %s. This may be due to your "+
			"system or onnxruntime library version not supporting OpenVINO.\n",
			e)
	}
	return options
}

func TestOpenVINOSession(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)
	sessionOptions := getOpenVINOSessionOptions(t)
	defer sessionOptions.Destroy()
	testBigSessionWithOptions(t, sessionOptions)
}

func BenchmarkOpenVINOSession(b *testing.B) {
	b.StopTimer()
	InitializeRuntime(b)
	defer CleanupRuntime(b)
	sessionOptions := getOpenVINOSessionOptions(b)
	defer sessionOptions.Destroy()
	benchmarkBigSessionWithOptions(b, sessionOptions)
}
