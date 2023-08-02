// This library wraps the C "onnxruntime" library maintained at
// https://github.com/microsoft/onnxruntime.  It seeks to provide as simple an
// interface as possible to load and run ONNX-format neural networks from
// Go code.
package onnxruntime_go

import (
	"fmt"
	"os"
	"unsafe"
)

// #cgo CFLAGS: -O2 -g
//
// #include "onnxruntime_wrapper.h"
import "C"

// This string should be the path to onnxruntime.so, or onnxruntime.dll.
var onnxSharedLibraryPath string

// For simplicity, this library maintains a single ORT environment internally.
var ortEnv *C.OrtEnv

// We also keep a single OrtMemoryInfo value around, since we only support CPU
// allocations for now.
var ortMemoryInfo *C.OrtMemoryInfo

var NotInitializedError error = fmt.Errorf("InitializeRuntime() has either " +
	"not yet been called, or did not return successfully")

var ZeroShapeLengthError error = fmt.Errorf("The shape has no dimensions")

var ShapeOverflowError error = fmt.Errorf("The shape's flattened size " +
	"overflows an int64")

// This type of error is returned when we attempt to validate a tensor that has
// a negative or 0 dimension.
type BadShapeDimensionError struct {
	DimensionIndex int
	DimensionSize  int64
}

func (e *BadShapeDimensionError) Error() string {
	return fmt.Sprintf("Dimension %d of the shape has invalid value %d",
		e.DimensionIndex, e.DimensionSize)
}

// Does two things: converts the given OrtStatus to a Go error, and releases
// the status. If the status is nil, this does nothing and returns nil.
func statusToError(status *C.OrtStatus) error {
	if status == nil {
		return nil
	}
	msg := C.GetErrorMessage(status)
	toReturn := C.GoString(msg)
	C.ReleaseOrtStatus(status)
	return fmt.Errorf("%s", toReturn)
}

// Use this function to set the path to the "onnxruntime.so" or
// "onnxruntime.dll" function. By default, it will be set to "onnxruntime.so"
// on non-Windows systems, and "onnxruntime.dll" on Windows. Users wishing to
// specify a particular location of this library must call this function prior
// to calling onnxruntime.InitializeEnvironment().
func SetSharedLibraryPath(path string) {
	onnxSharedLibraryPath = path
}

// Returns false if the onnxruntime package is not initialized. Called
// internally by several functions, to avoid segfaulting if
// InitializeEnvironment hasn't been called yet.
func IsInitialized() bool {
	return ortEnv != nil
}

// Call this function to initialize the internal onnxruntime environment. If
// this doesn't return an error, the caller will be responsible for calling
// DestroyEnvironment to free the onnxruntime state when no longer needed.
func InitializeEnvironment() error {
	if IsInitialized() {
		return fmt.Errorf("The onnxruntime has already been initialized")
	}
	// Do the windows- or linux- specific initialization first.
	e := platformInitializeEnvironment()
	if e != nil {
		return fmt.Errorf("Platform-specific initialization failed: %w", e)
	}

	name := C.CString("Golang onnxruntime environment")
	defer C.free(unsafe.Pointer(name))
	status := C.CreateOrtEnv(name, &ortEnv)
	if status != nil {
		return fmt.Errorf("Error creating ORT environment: %w",
			statusToError(status))
	}

	status = C.CreateOrtMemoryInfo(&ortMemoryInfo)
	if status != nil {
		DestroyEnvironment()
		return fmt.Errorf("Error creating ORT memory info: %w",
			statusToError(status))
	}

	return nil
}

// Call this function to cleanup the internal onnxruntime environment when it
// is no longer needed.
func DestroyEnvironment() error {
	var e error
	if !IsInitialized() {
		return NotInitializedError
	}
	if ortMemoryInfo != nil {
		C.ReleaseOrtMemoryInfo(ortMemoryInfo)
		ortMemoryInfo = nil
	}
	if ortEnv != nil {
		C.ReleaseOrtEnv(ortEnv)
		ortEnv = nil
	}

	// platformCleanup primarily unloads the library, so we need to call it
	// last, after any functions that make use of the ORT API.
	e = platformCleanup()
	if e != nil {
		return fmt.Errorf("Platform-specific cleanup failed: %w", e)
	}
	return nil
}

// Disables telemetry events for the onnxruntime environment. Must be called
// after initializing the environment using InitializeEnvironment(). It is
// unclear from the onnxruntime docs whether this will cause an error or
// silently return if telemetry is already disabled.
func DisableTelemetry() error {
	if !IsInitialized() {
		return NotInitializedError
	}
	status := C.DisableTelemetry(ortEnv)
	if status != nil {
		return fmt.Errorf("Error disabling onnxruntime telemetry: %w",
			statusToError(status))
	}
	return nil
}

// Enables telemetry events for the onnxruntime environment. Must be called
// after initializing the environment using InitializeEnvironment(). It is
// unclear from the onnxruntime docs whether this will cause an error or
// silently return if telemetry is already enabled.
func EnableTelemetry() error {
	if !IsInitialized() {
		return NotInitializedError
	}
	status := C.EnableTelemetry(ortEnv)
	if status != nil {
		return fmt.Errorf("Error enabling onnxruntime telemetry: %w",
			statusToError(status))
	}
	return nil
}

// The Shape type holds the shape of the tensors used by the network input and
// outputs.
type Shape []int64

// Returns a Shape, with the given dimensions.
func NewShape(dimensions ...int64) Shape {
	return Shape(dimensions)
}

// Returns the total number of elements in a tensor with the given shape. Note
// that this may be an invalid value due to overflow or negative dimensions. If
// a shape comes from an untrusted source, it may be a good practice to call
// Validate() prior to trusting the FlattenedSize.
func (s Shape) FlattenedSize() int64 {
	if len(s) == 0 {
		return 0
	}
	toReturn := int64(s[0])
	for i := 1; i < len(s); i++ {
		toReturn *= s[i]
	}
	return toReturn
}

// Returns a non-nil error if the shape has bad or zero dimensions. May return
// a ZeroShapeLengthError, a ShapeOverflowError, or an BadShapeDimensionError.
// In the future, this may return other types of errors if it others become
// necessary.
func (s Shape) Validate() error {
	if len(s) == 0 {
		return ZeroShapeLengthError
	}
	if s[0] <= 0 {
		return &BadShapeDimensionError{
			DimensionIndex: 0,
			DimensionSize:  s[0],
		}
	}
	flattenedSize := int64(s[0])
	for i := 1; i < len(s); i++ {
		d := s[i]
		if d <= 0 {
			return &BadShapeDimensionError{
				DimensionIndex: i,
				DimensionSize:  d,
			}
		}
		tmp := flattenedSize * d
		if tmp < flattenedSize {
			return ShapeOverflowError
		}
		flattenedSize = tmp
	}
	return nil
}

// Makes and returns a deep copy of the Shape.
func (s Shape) Clone() Shape {
	toReturn := make([]int64, len(s))
	copy(toReturn, []int64(s))
	return Shape(toReturn)
}

func (s Shape) String() string {
	return fmt.Sprintf("%v", []int64(s))
}

// Returns true if both shapes match in every dimension.
func (s Shape) Equals(other Shape) bool {
	if len(s) != len(other) {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] != other[i] {
			return false
		}
	}
	return true
}

type Tensor[T TensorData] struct {
	// The shape of the tensor
	shape Shape
	// The go slice containing the flattened data that backs the ONNX tensor.
	data []T
	// The underlying ONNX value we use with the C API.
	ortValue *C.OrtValue
}

// Cleans up and frees the memory associated with this tensor.
func (t *Tensor[_]) Destroy() error {
	C.ReleaseOrtValue(t.ortValue)
	t.ortValue = nil
	t.data = nil
	t.shape = nil
	return nil
}

// Returns the slice containing the tensor's underlying data. The contents of
// the slice can be read or written to get or set the tensor's contents.
func (t *Tensor[T]) GetData() []T {
	return t.data
}

// Returns the shape of the tensor. The returned shape is only a copy;
// modifying this does *not* change the shape of the underlying tensor.
// (Modifying the tensor's shape can only be accomplished by Destroying and
// recreating the tensor with the same data.)
func (t *Tensor[_]) GetShape() Shape {
	return t.shape.Clone()
}

// Makes a deep copy of the tensor, including its ONNXRuntime value. The Tensor
// returned by this function must be destroyed when no longer needed. The
// returned tensor will also no longer refer to the same underlying data; use
// GetData() to obtain the new underlying slice.
func (t *Tensor[T]) Clone() (*Tensor[T], error) {
	toReturn, e := NewEmptyTensor[T](t.shape)
	if e != nil {
		return nil, fmt.Errorf("Error allocating tensor clone: %w", e)
	}
	copy(toReturn.GetData(), t.data)
	return toReturn, nil
}

// Creates a new empty tensor with the given shape. The shape provided to this
// function is copied, and is no longer needed after this function returns.
func NewEmptyTensor[T TensorData](s Shape) (*Tensor[T], error) {
	e := s.Validate()
	if e != nil {
		return nil, fmt.Errorf("Invalid tensor shape: %w", e)
	}
	elementCount := s.FlattenedSize()
	data := make([]T, elementCount)
	return NewTensor(s, data)
}

// Creates a new tensor backed by an existing data slice. The shape provided to
// this function is copied, and is no longer needed after this function
// returns. If the data slice is longer than s.FlattenedSize(), then only the
// first portion of the data will be used.
func NewTensor[T TensorData](s Shape, data []T) (*Tensor[T], error) {
	if !IsInitialized() {
		return nil, NotInitializedError
	}
	e := s.Validate()
	if e != nil {
		return nil, fmt.Errorf("Invalid tensor shape: %w", e)
	}
	elementCount := s.FlattenedSize()
	if elementCount > int64(len(data)) {
		return nil, fmt.Errorf("The tensor's shape (%s) requires %d "+
			"elements, but only %d were provided\n", s, elementCount,
			len(data))
	}
	var ortValue *C.OrtValue
	dataType := GetTensorElementDataType[T]()
	dataSize := unsafe.Sizeof(data[0]) * uintptr(elementCount)

	status := C.CreateOrtTensorWithShape(unsafe.Pointer(&data[0]),
		C.size_t(dataSize), (*C.int64_t)(unsafe.Pointer(&s[0])),
		C.int64_t(len(s)), ortMemoryInfo, dataType, &ortValue)
	if status != nil {
		return nil, fmt.Errorf("ORT API error creating tensor: %s",
			statusToError(status))
	}

	toReturn := Tensor[T]{
		data:     data[0:elementCount],
		shape:    s.Clone(),
		ortValue: ortValue,
	}
	// TODO: Set a finalizer on new Tensors to hopefully prevent careless
	// memory leaks.
	// - Idea: use a "destroyable" interface?
	return &toReturn, nil
}

// A wrapper around the OrtSession C struct. Requires the user to maintain all
// input and output tensors, and to use the same data type for input and output
// tensors.
type Session[T TensorData] struct {
	ortSession *C.OrtSession
	// We convert the tensor names to C strings only once, and keep them around
	// here for future calls to Run().
	inputNames  []*C.char
	outputNames []*C.char
	// We only actually keep around the OrtValue pointers from the tensors.
	inputs  []*C.OrtValue
	outputs []*C.OrtValue
}

// Similar to Session, but does not require the specification of the input and output
// shapes at session creation time, and allows for input and output tensors to have
// different types. This allows for fully dynamic input to the onnx model.
type DynamicSession[In TensorData, Out TensorData] struct {
	ortSession  *C.OrtSession
	inputNames  []*C.char
	outputNames []*C.char
}

// The same as NewSession, but takes a slice of bytes containing the .onnx
// network rather than a file path.
func NewSessionWithONNXData[T TensorData](onnxData []byte, inputNames,
	outputNames []string, inputs, outputs []*Tensor[T]) (*Session[T], error) {
	if !IsInitialized() {
		return nil, NotInitializedError
	}
	if len(inputs) == 0 {
		return nil, fmt.Errorf("No inputs were provided")
	}
	if len(outputs) == 0 {
		return nil, fmt.Errorf("No outputs were provided")
	}
	if len(inputs) != len(inputNames) {
		return nil, fmt.Errorf("Got %d input tensors, but %d input names",
			len(inputs), len(inputNames))
	}
	if len(outputs) != len(outputNames) {
		return nil, fmt.Errorf("Got %d output tensors, but %d output names",
			len(outputs), len(outputNames))
	}

	var ortSession *C.OrtSession
	status := C.CreateSession(unsafe.Pointer(&(onnxData[0])),
		C.size_t(len(onnxData)), ortEnv, &ortSession)
	if status != nil {
		return nil, fmt.Errorf("Error creating session: %w",
			statusToError(status))
	}
	// ONNXRuntime copies the file content unless a specific flag is provided
	// when creating the session (and we don't provide it!)

	// Collect the inputs and outputs, along with their names, into a format
	// more convenient for passing to the Run() function in the C API.
	cInputNames := make([]*C.char, len(inputNames))
	cOutputNames := make([]*C.char, len(outputNames))
	for i, v := range inputNames {
		cInputNames[i] = C.CString(v)
	}
	for i, v := range outputNames {
		cOutputNames[i] = C.CString(v)
	}
	inputOrtTensors := make([]*C.OrtValue, len(inputs))
	outputOrtTensors := make([]*C.OrtValue, len(outputs))
	for i, v := range inputs {
		inputOrtTensors[i] = v.ortValue
	}
	for i, v := range outputs {
		outputOrtTensors[i] = v.ortValue
	}
	return &Session[T]{
		ortSession:  ortSession,
		inputNames:  cInputNames,
		outputNames: cOutputNames,
		inputs:      inputOrtTensors,
		outputs:     outputOrtTensors,
	}, nil
}

// Similar to NewSessionWithOnnxData, but for dynamic sessions.
func NewDynamicSessionWithONNXData[in TensorData, out TensorData](onnxData []byte, inputNames, outputNames []string) (*DynamicSession[in, out], error) {
	var ortSession *C.OrtSession
	status := C.CreateSession(unsafe.Pointer(&(onnxData[0])),
		C.size_t(len(onnxData)), ortEnv, &ortSession)
	if status != nil {
		return nil, fmt.Errorf("Error creating session: %w",
			statusToError(status))
	}
	// ONNXRuntime copies the file content unless a specific flag is provided
	// when creating the session (and we don't provide it!)
	cInputNames := make([]*C.char, len(inputNames))
	cOutputNames := make([]*C.char, len(outputNames))
	for i, v := range inputNames {
		cInputNames[i] = C.CString(v)
	}
	for i, v := range outputNames {
		cOutputNames[i] = C.CString(v)
	}
	return &DynamicSession[in, out]{
		ortSession:  ortSession,
		inputNames:  cInputNames,
		outputNames: cOutputNames,
	}, nil

}

// Loads the ONNX network at the given path, and initializes a Session
// instance. If this returns successfully, the caller must call Destroy() on
// the returned session when it is no longer needed. We require the user to
// provide the input and output tensors and names at this point, in order to
// not need to re-allocate them every time Run() is called. The user instead
// can just update or access the input/output tensor data after calling Run().
// The input and output tensors MUST outlive this session, and calling
// session.Destroy() will not destroy the input or output tensors.
func NewSession[T TensorData](onnxFilePath string, inputNames,
	outputNames []string, inputs, outputs []*Tensor[T]) (*Session[T], error) {
	fileContent, e := os.ReadFile(onnxFilePath)
	if e != nil {
		return nil, fmt.Errorf("Error reading %s: %w", onnxFilePath, e)
	}

	toReturn, e := NewSessionWithONNXData[T](fileContent, inputNames,
		outputNames, inputs, outputs)
	if e != nil {
		return nil, fmt.Errorf("Error creating session from %s: %w",
			onnxFilePath, e)
	}
	return toReturn, nil
}

// Same as NewSession, but for dynamic sessions.
func NewDynamicSession[in TensorData, out TensorData](onnxFilePath string, inputNames, outputNames []string) (*DynamicSession[in, out], error) {
	fileContent, e := os.ReadFile(onnxFilePath)
	if e != nil {
		return nil, fmt.Errorf("Error reading %s: %w", onnxFilePath, e)
	}

	toReturn, e := NewDynamicSessionWithONNXData[in, out](fileContent, inputNames, outputNames)
	if e != nil {
		return nil, fmt.Errorf("Error creating session from %s: %w",
			onnxFilePath, e)
	}
	return toReturn, nil
}

func (s *Session[_]) Destroy() error {
	if s.ortSession != nil {
		C.ReleaseOrtSession(s.ortSession)
		s.ortSession = nil
	}
	for i := range s.inputNames {
		C.free(unsafe.Pointer(s.inputNames[i]))
	}
	s.inputNames = nil
	for i := range s.outputNames {
		C.free(unsafe.Pointer(s.outputNames[i]))
	}
	s.outputNames = nil
	s.inputs = nil
	s.outputs = nil
	return nil
}

func (s *DynamicSession[_, _]) Destroy() error {
	if s.ortSession != nil {
		C.ReleaseOrtSession(s.ortSession)
		s.ortSession = nil
	}
	for i := range s.inputNames {
		C.free(unsafe.Pointer(s.inputNames[i]))
	}
	s.inputNames = nil
	for i := range s.outputNames {
		C.free(unsafe.Pointer(s.outputNames[i]))
	}
	s.outputNames = nil
	return nil
}

// Runs the session, updating the contents of the output tensors on success.
func (s *Session[T]) Run() error {
	status := C.RunOrtSession(s.ortSession, &s.inputs[0], &s.inputNames[0],
		C.int(len(s.inputs)), &s.outputs[0], &s.outputNames[0],
		C.int(len(s.outputs)))
	if status != nil {
		return fmt.Errorf("Error running network: %w", statusToError(status))
	}
	return nil
}

// Runs the dynamic session. Differently from the Session object, this method
// requires the caller to provide the slice of input and output tensor pointer
// of the right type. The resulting output is stored in the output tensors, and
// it is the responsibility of the caller to destroy the tensors to free memory.
func (s *DynamicSession[in, out]) Run(inputs []*Tensor[in], outputs []*Tensor[out]) error {
	// create inputs
	// destroying the input and output tensors will also destroy the ort values
	inputOrtTensors := make([]*C.OrtValue, len(inputs))
	for i, v := range inputs {
		inputOrtTensors[i] = v.ortValue
	}
	outputOrtTensors := make([]*C.OrtValue, len(outputs))
	for i, v := range outputs {
		outputOrtTensors[i] = v.ortValue
	}
	status := C.RunOrtSession(s.ortSession, &inputOrtTensors[0], &s.inputNames[0],
		C.int(len(inputs)), &outputOrtTensors[0], &s.outputNames[0],
		C.int(len(outputs)))
	if status != nil {
		return fmt.Errorf("Error running network: %w", statusToError(status))
	}
	return nil
}
