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
// a ZeroShapeLengthError, a ShapeOverflowError, or a BadShapeDimensionError.
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

// This wraps internal implementation details to avoid exposing them to users
// via the ArbitraryTensor interface.
type TensorInternalData struct {
	ortValue *C.OrtValue
}

// An interface for managing tensors where we don't care about accessing the
// underlying data slice. All typed tensors will support this interface,
// regardless of the underlying data type.
type ArbitraryTensor interface {
	DataType() C.ONNXTensorElementDataType
	GetShape() Shape
	Destroy() error
	GetInternals() *TensorInternalData
}

// Used to manage all input and output data for onnxruntime networks. A Tensor
// always has an associated type and refers to data contained in an underlying
// Go slice. New tensors should be created using the NewTensor or
// NewEmptyTensor functions, and must be destroyed using the Destroy function
// when no longer needed.
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

// Returns the value from the ONNXTensorElementDataType C enum corresponding to
// the type of data held by this tensor.
func (t *Tensor[T]) DataType() C.ONNXTensorElementDataType {
	return GetTensorElementDataType[T]()
}

// Returns the shape of the tensor. The returned shape is only a copy;
// modifying this does *not* change the shape of the underlying tensor.
// (Modifying the tensor's shape can only be accomplished by Destroying and
// recreating the tensor with the same data.)
func (t *Tensor[_]) GetShape() Shape {
	return t.shape.Clone()
}

func (t *Tensor[_]) GetInternals() *TensorInternalData {
	return &TensorInternalData{
		ortValue: t.ortValue,
	}
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

// Holds options required when enabling the CUDA backend for a session. This
// struct wraps C onnxruntime types; users must create instances of this using
// the NewCUDAProviderOptions() function. So, to enable CUDA for a session,
// follow these steps:
//
//  1. Call NewSessionOptions() to create a SessionOptions struct.
//  2. Call NewCUDAProviderOptions() to obtain a CUDAProviderOptions struct.
//  3. Call the CUDAProviderOptions struct's Update(...) function to pass a
//     list of settings to CUDA. (See the comment on the Update() function.)
//  4. Pass the CUDA options struct pointer to the
//     SessionOptions.AppendExecutionProviderCUDA(...) function.
//  5. Call the Destroy() function on the CUDA provider options.
//  6. Call NewAdvancedSession(...), passing the SessionOptions struct to it.
//  7. Call the Destroy() function on the SessionOptions struct.
//
// Admittedly, this is a bit of a mess, but that's how it's handled by the C
// API internally. (The onnxruntime python API hides a bunch of this complexity
// using getter and setter functions, for which Go does not have a terse
// equivalent.)
type CUDAProviderOptions struct {
	o *C.OrtCUDAProviderOptionsV2
}

// Used when setting key-value pair options with certain obnoxious C APIs.
// The entries in each of the returned slices must be freed when they're
// no longer needed.
func mapToCStrings(options map[string]string) ([]*C.char, []*C.char) {
	keys := make([]*C.char, 0, len(options))
	values := make([]*C.char, 0, len(options))
	for k, v := range options {
		keys = append(keys, C.CString(k))
		values = append(values, C.CString(v))
	}
	return keys, values
}

// Calls free on each entry in the array of C strings.
func freeCStrings(s []*C.char) {
	for i := range s {
		C.free(unsafe.Pointer(s[i]))
		s[i] = nil
	}
}

// Wraps the call to the UpdateCUDAProviderOptions in the onnxruntime C API.
// Requires a map of string keys to values for configuring the CUDA backend.
// For example, set the key "device_id" to "1" to use GPU 1 rather than 0.
//
// The onnxruntime headers refer users to
// https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#configuration-options
// for a full list of available keys and values.
func (o *CUDAProviderOptions) Update(options map[string]string) error {
	if len(options) == 0 {
		return nil
	}
	keys, values := mapToCStrings(options)
	defer freeCStrings(keys)
	defer freeCStrings(values)
	status := C.UpdateCUDAProviderOptions(o.o, &(keys[0]), &(values[0]),
		C.int(len(options)))
	if status != nil {
		return statusToError(status)
	}
	return nil
}

// Must be called when the CUDAProviderOptions struct is no longer needed;
// frees internal C-allocated state. Note that the CUDAProviderOptions struct
// can be destroyed as soon as options.AppendExecutionProviderCUDA has been
// called.
func (o *CUDAProviderOptions) Destroy() error {
	if o.o == nil {
		return fmt.Errorf("The CUDAProviderOptions are not initialized")
	}
	C.ReleaseCUDAProviderOptions(o.o)
	o.o = nil
	return nil
}

// Initializes and returns a CUDAProviderOptions struct, used when enabling
// CUDA in a SessionOptions instance. (i.e., a CUDAProviderOptions must be
// configured, then passed to SessionOptions.AppendExecutionProviderCUDA.)
// The caller must call the Destroy() function on the returned struct when it's
// no longer needed.
func NewCUDAProviderOptions() (*CUDAProviderOptions, error) {
	if !IsInitialized() {
		return nil, NotInitializedError
	}
	var o *C.OrtCUDAProviderOptionsV2
	status := C.CreateCUDAProviderOptions(&o)
	if status != nil {
		return nil, statusToError(status)
	}
	return &CUDAProviderOptions{
		o: o,
	}, nil
}

// Like the CUDAProviderOptions struct, but used for configuring TensorRT
// options. Instances of this struct must be initialized using
// NewTensorRTProviderOptions() and cleaned up by calling their Destroy()
// function when they are no longer needed.
type TensorRTProviderOptions struct {
	o *C.OrtTensorRTProviderOptionsV2
}

// Wraps the call to the UpdateTensorRTProviderOptions in the C API. Requires
// a map of string keys to values.
//
// The onnxruntime headers refer users to
// https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#cc
// for the list of available keys and values.
func (o *TensorRTProviderOptions) Update(options map[string]string) error {
	if len(options) == 0 {
		return nil
	}
	keys, values := mapToCStrings(options)
	defer freeCStrings(keys)
	defer freeCStrings(values)
	status := C.UpdateTensorRTProviderOptions(o.o, &(keys[0]), &(values[0]),
		C.int(len(options)))
	if status != nil {
		return statusToError(status)
	}
	return nil
}

// Must be called when the TensorRTProviderOptions are no longer needed, in
// order to free internal state. The struct is not needed as soon as you have
// passed it to the AppendExecutionProviderTensorRT function.
func (o *TensorRTProviderOptions) Destroy() error {
	if o.o == nil {
		return fmt.Errorf("The TensorRTProviderOptions are not initialized")
	}
	C.ReleaseTensorRTProviderOptions(o.o)
	o.o = nil
	return nil
}

// Initializes and returns a TensorRTProviderOptions struct, used when enabling
// the TensorRT backend. The caller must call the Destroy() function on the
// returned struct when it's no longer needed.
func NewTensorRTProviderOptions() (*TensorRTProviderOptions, error) {
	if !IsInitialized() {
		return nil, NotInitializedError
	}
	var o *C.OrtTensorRTProviderOptionsV2
	status := C.CreateTensorRTProviderOptions(&o)
	if status != nil {
		return nil, statusToError(status)
	}
	return &TensorRTProviderOptions{
		o: o,
	}, nil
}

// Used to set options when creating an ONNXRuntime session. There is currently
// not a way to change options after the session is created, apart from
// destroying the session and creating a new one. This struct opaquely wraps a
// C OrtSessionOptions struct, which users must modify via function calls. (The
// OrtSessionOptions struct is opaque in the C API, too.)
//
// Users must instantiate this struct using the NewSessionOptions function.
// Instances must be destroyed by calling the Destroy() method after the
// options are no longer needed (after NewAdvancedSession(...) has returned).
type SessionOptions struct {
	o *C.OrtSessionOptions
}

func (o *SessionOptions) Destroy() error {
	if o.o == nil {
		return fmt.Errorf("The SessionOptions are not initialized")
	}
	C.ReleaseSessionOptions(o.o)
	o.o = nil
	return nil
}

// Sets the number of threads used to parallelize execution within onnxruntime
// graph nodes. A value of 0 uses the default number of threads.
func (o *SessionOptions) SetIntraOpNumThreads(n int) error {
	if n < 0 {
		return fmt.Errorf("Number of threads must be at least 0, got %d", n)
	}
	status := C.SetIntraOpNumThreads(o.o, C.int(n))
	if status != nil {
		return statusToError(status)
	}
	return nil
}

// Sets the number of threads used to parallelize execution across separate
// onnxruntime graph nodes. A value of 0 uses the default number of threads.
func (o *SessionOptions) SetInterOpNumThreads(n int) error {
	if n < 0 {
		return fmt.Errorf("Number of threads must be at least 0, got %d", n)
	}
	status := C.SetInterOpNumThreads(o.o, C.int(n))
	if status != nil {
		return statusToError(status)
	}
	return nil
}

// Takes a pointer to an initialized CUDAProviderOptions instance, and applies
// them to the session options. This is what you'll need to call if you want
// the session to use CUDA. Returns an error if your device (or onnxruntime
// library) does not support CUDA. The CUDAProviderOptions struct can be
// destroyed after this.
func (o *SessionOptions) AppendExecutionProviderCUDA(
	cudaOptions *CUDAProviderOptions) error {
	status := C.AppendExecutionProviderCUDAV2(o.o, cudaOptions.o)
	if status != nil {
		return statusToError(status)
	}
	return nil
}

// Takes an initialized TensorRTProviderOptions instance, and applies them to
// the session options. You'll need to call this if you want the session to use
// TensorRT. Returns an error if your device (or onnxruntime library version)
// does not support TensorRT. The TensorRTProviderOptions can be destroyed
// after this.
func (o *SessionOptions) AppendExecutionProviderTensorRT(
	tensorRTOptions *TensorRTProviderOptions) error {
	status := C.AppendExecutionProviderTensorRTV2(o.o, tensorRTOptions.o)
	if status != nil {
		return statusToError(status)
	}
	return nil
}

// Enables the CoreML backend for the given session options on supported
// platforms. Unlike the other AppendExecutionProvider* functions, this one
// only takes a bitfield of flags rather than an options object, though it
// wouldn't suprise me if onnxruntime deprecated this API in the future as it
// did with the others. If that happens, we'll likely add a
// CoreMLProviderOptions struct and an AppendExecutionProviderCoreMLV2 function
// to the Go wrapper library, but for now the simpler API is the only thing
// available.
//
// Regardless, the meanings of the flag bits are currently defined in the
// coreml_provider_factory.h file which is provided in the include/ directory of
// the onnxruntime releases for Apple platforms.
func (o *SessionOptions) AppendExecutionProviderCoreML(flags uint32) error {
	status := C.AppendExecutionProviderCoreML(o.o, C.uint32_t(flags))
	if status != nil {
		return statusToError(status)
	}
	return nil
}

// Initializes and returns a SessionOptions struct, used when setting options
// in new AdvancedSession instances. The caller must call the Destroy()
// function on the returned struct when it's no longer needed.
func NewSessionOptions() (*SessionOptions, error) {
	if !IsInitialized() {
		return nil, NotInitializedError
	}
	var o *C.OrtSessionOptions
	status := C.CreateSessionOptions(&o)
	if status != nil {
		return nil, statusToError(status)
	}
	return &SessionOptions{
		o: o,
	}, nil
}

// A wrapper around the OrtSession C struct. Requires the user to maintain all
// input and output tensors, and to use the same data type for input and output
// tensors. Created using NewAdvancedSession(...) or
// NewAdvancedSessionWithONNXData(...). The caller is responsible for calling
// the Destroy() function on each session when it is no longer needed.
type AdvancedSession struct {
	ortSession *C.OrtSession
	// We convert the tensor names to C strings only once, and keep them around
	// here for future calls to Run().
	inputNames  []*C.char
	outputNames []*C.char
	// We only need the OrtValue pointers from the tensors when working with
	// the C API. Also, these fields aren't used with a DynamicAdvancedSession.
	inputs  []*C.OrtValue
	outputs []*C.OrtValue
}

func createCSession(onnxData []byte, options *SessionOptions) (*C.OrtSession,
	error) {
	if !IsInitialized() {
		return nil, NotInitializedError
	}
	if len(onnxData) == 0 {
		return nil, fmt.Errorf("Missing ONNX data")
	}
	var ortSession *C.OrtSession
	var ortSessionOptions *C.OrtSessionOptions
	if options != nil {
		ortSessionOptions = options.o
	}
	status := C.CreateSession(unsafe.Pointer(&(onnxData[0])),
		C.size_t(len(onnxData)), ortEnv, &ortSession, ortSessionOptions)
	if status != nil {
		return nil, statusToError(status)
	}
	return ortSession, nil
}

// The same as NewAdvancedSession, but takes a slice of bytes containing the
// .onnx network rather than a file path.
func NewAdvancedSessionWithONNXData(onnxData []byte, inputNames,
	outputNames []string, inputs, outputs []ArbitraryTensor,
	options *SessionOptions) (*AdvancedSession, error) {
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

	ortSession, e := createCSession(onnxData, options)
	if e != nil {
		return nil, fmt.Errorf("Error creating C session: %w", e)
	}

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
		inputOrtTensors[i] = v.GetInternals().ortValue
	}
	for i, v := range outputs {
		outputOrtTensors[i] = v.GetInternals().ortValue
	}
	return &AdvancedSession{
		ortSession:  ortSession,
		inputNames:  cInputNames,
		outputNames: cOutputNames,
		inputs:      inputOrtTensors,
		outputs:     outputOrtTensors,
	}, nil
}

// Loads the ONNX network at the given path, and initializes an AdvancedSession
// instance. If this returns successfully, the caller must call Destroy() on
// the returned session when it is no longer needed. We require the user to
// provide the input and output tensors and names at this point, in order to
// not need to re-allocate them every time Run() is called. The user instead
// can just update or access the input/output tensor data after calling Run().
// The input and output tensors MUST outlive this session, and calling
// session.Destroy() will not destroy the input or output tensors. If the
// provided SessionOptions pointer is nil, then the new session will use
// default options.
func NewAdvancedSession(onnxFilePath string, inputNames, outputNames []string,
	inputs, outputs []ArbitraryTensor,
	options *SessionOptions) (*AdvancedSession, error) {
	fileContent, e := os.ReadFile(onnxFilePath)
	if e != nil {
		return nil, fmt.Errorf("Error reading %s: %w", onnxFilePath, e)
	}
	toReturn, e := NewAdvancedSessionWithONNXData(fileContent, inputNames,
		outputNames, inputs, outputs, options)
	if e != nil {
		return nil, fmt.Errorf("Error creating session from %s: %w",
			onnxFilePath, e)
	}
	return toReturn, nil
}

func (s *AdvancedSession) Destroy() error {
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

// Runs the session, updating the contents of the output tensors on success.
func (s *AdvancedSession) Run() error {
	status := C.RunOrtSession(s.ortSession, &s.inputs[0], &s.inputNames[0],
		C.int(len(s.inputs)), &s.outputs[0], &s.outputNames[0],
		C.int(len(s.outputs)))
	if status != nil {
		return fmt.Errorf("Error running network: %w", statusToError(status))
	}
	return nil
}

// This type of session does not require specifying input and output tensors
// ahead of time, but allows users to pass the list of input and output tensors
// when calling Run(). As with AdvancedSession, users must still call Destroy()
// on an DynamicAdvancedSession that is no longer needed.
type DynamicAdvancedSession struct {
	// We may have further performance optimizations to this in the future, but
	// for now it's just a regular AdvancedSession.
	s *AdvancedSession
}

// Like NewAdvancedSessionWithONNXData, but does not require specifying input
// and output tensors.
func NewDynamicAdvancedSessionWithONNXData(onnxData []byte,
	inputNames, outputNames []string,
	options *SessionOptions) (*DynamicAdvancedSession, error) {
	ortSession, e := createCSession(onnxData, options)
	if e != nil {
		return nil, fmt.Errorf("Error creating C session: %w", e)
	}
	cInputNames := make([]*C.char, len(inputNames))
	cOutputNames := make([]*C.char, len(outputNames))
	for i, v := range inputNames {
		cInputNames[i] = C.CString(v)
	}
	for i, v := range outputNames {
		cOutputNames[i] = C.CString(v)
	}
	// We don't use the input and output list of OrtValues with these.
	s := &AdvancedSession{
		ortSession:  ortSession,
		inputNames:  cInputNames,
		outputNames: cOutputNames,
		inputs:      nil,
		outputs:     nil,
	}
	return &DynamicAdvancedSession{
		s: s,
	}, nil
}

// Like NewAdvancedSession, but does not require specifying input and output
// tensors.
func NewDynamicAdvancedSession(onnxFilePath string, inputNames,
	outputNames []string, options *SessionOptions) (*DynamicAdvancedSession,
	error) {
	fileContent, e := os.ReadFile(onnxFilePath)
	if e != nil {
		return nil, fmt.Errorf("Error reading %s: %w", onnxFilePath, e)
	}

	toReturn, e := NewDynamicAdvancedSessionWithONNXData(fileContent,
		inputNames, outputNames, options)
	if e != nil {
		return nil, fmt.Errorf("Error creating dynamic session from %s: %w",
			onnxFilePath, e)
	}
	return toReturn, nil
}

func (s *DynamicAdvancedSession) Destroy() error {
	return s.s.Destroy()
}

// Runs the network on the given input and output tensors. The number of input
// and output tensors must match the number (and order) of the input and output
// names specified to NewDynamicAdvancedSession.
func (s *DynamicAdvancedSession) Run(inputs, outputs []ArbitraryTensor) error {
	if len(inputs) != len(s.s.inputNames) {
		return fmt.Errorf("The session specified %d input names, but Run() "+
			"was called with %d input tensors", len(s.s.inputNames),
			len(inputs))
	}
	if len(outputs) != len(s.s.outputNames) {
		return fmt.Errorf("The session specified %d output names, but Run() "+
			"was called with %d output tensors", len(s.s.outputNames),
			len(outputs))
	}
	inputValues := make([]*C.OrtValue, len(inputs))
	for i, v := range inputs {
		inputValues[i] = v.GetInternals().ortValue
	}
	outputValues := make([]*C.OrtValue, len(outputs))
	for i, v := range outputs {
		outputValues[i] = v.GetInternals().ortValue
	}
	status := C.RunOrtSession(s.s.ortSession, &inputValues[0],
		&s.s.inputNames[0], C.int(len(inputs)), &outputValues[0],
		&s.s.outputNames[0], C.int(len(outputs)))
	if status != nil {
		return fmt.Errorf("Error running network: %w", statusToError(status))
	}
	return nil
}
