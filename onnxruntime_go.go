// This library wraps the C "onnxruntime" library maintained at
// https://github.com/microsoft/onnxruntime.  It seeks to provide as simple an
// interface as possible to load and run ONNX-format neural networks from
// Go code.
package onnxruntime_go

import (
	"fmt"
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

// GetVersion return version of the Onnxruntime library for logging.
func GetVersion() string {
	return C.GoString(C.GetVersion())
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

	// Get the training API pointer if it is supported.
	C.SetTrainingApi()

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
// via the Value interface.
type ValueInternalData struct {
	ortValue *C.OrtValue
}

// An interface for managing tensors or other onnxruntime values where we don't
// necessarily need to access the underlying data slice. All typed tensors will
// support this interface regardless of the underlying data type.
type Value interface {
	DataType() C.ONNXTensorElementDataType
	GetShape() Shape
	Destroy() error
	GetInternals() *ValueInternalData
	ZeroContents()
	GetONNXType() ONNXType
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
	// The number of bytes taken by the data slice.
	dataSize uintptr
	// The underlying ONNX value we use with the C API.
	ortValue *C.OrtValue
}

// Cleans up and frees the memory associated with this tensor.
func (t *Tensor[_]) Destroy() error {
	C.ReleaseOrtValue(t.ortValue)
	t.ortValue = nil
	t.data = nil
	t.dataSize = 0
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
//
// NOTE: This function was added prior to the introduction of the
// Go TensorElementDataType int wrapping the C enum, so it still returns the
// CGo type.
func (t *Tensor[T]) DataType() C.ONNXTensorElementDataType {
	return GetTensorElementDataType[T]()
}

// Always returns ONNXTypeTensor for any Tensor[T] even if the underlying
// tensor is invalid for some reason.
func (t *Tensor[_]) GetONNXType() ONNXType {
	return ONNXTypeTensor
}

// Returns the shape of the tensor. The returned shape is only a copy;
// modifying this does *not* change the shape of the underlying tensor.
// (Modifying the tensor's shape can only be accomplished by Destroying and
// recreating the tensor with the same data.)
func (t *Tensor[_]) GetShape() Shape {
	return t.shape.Clone()
}

func (t *Tensor[_]) GetInternals() *ValueInternalData {
	return &ValueInternalData{
		ortValue: t.ortValue,
	}
}

// Sets every element in the tensor's underlying data slice to 0.
func (t *Tensor[T]) ZeroContents() {
	C.memset(unsafe.Pointer(&t.data[0]), 0, C.size_t(t.dataSize))
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
			"elements, but only %d were provided", s, elementCount,
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
		dataSize: dataSize,
		shape:    s.Clone(),
		ortValue: ortValue,
	}
	// TODO: Set a finalizer on new Tensors to hopefully prevent careless
	// memory leaks.
	// - Idea: use a "destroyable" interface?
	return &toReturn, nil
}

// Wraps the ONNXTEnsorElementDataType enum in C.
type TensorElementDataType int

const (
	TensorElementDataTypeUndefined = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
	TensorElementDataTypeFloat     = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
	TensorElementDataTypeUint8     = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
	TensorElementDataTypeInt8      = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
	TensorElementDataTypeUint16    = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
	TensorElementDataTypeInt16     = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
	TensorElementDataTypeInt32     = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
	TensorElementDataTypeInt64     = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
	TensorElementDataTypeString    = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING
	TensorElementDataTypeBool      = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
	TensorElementDataTypeFloat16   = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
	TensorElementDataTypeDouble    = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
	TensorElementDataTypeUint32    = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
	TensorElementDataTypeUint64    = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64

	// Not supported by onnxruntime (as of onnxruntime version 1.19.0)
	TensorElementDataTypeComplex64 = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64
	// Not supported by onnxruntime (as of onnxruntime version 1.19.0)
	TensorElementDataTypeComplex128 = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128

	// Non-IEEE floating-point format based on IEEE754 single-precision
	TensorElementDataTypeBFloat16 = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16

	// 8-bit float types, introduced in onnx 1.14.  See
	// https://onnx.ai/onnx/technical/float8.html
	TensorElementDataTypeFloat8E4M3FN   = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN
	TensorElementDataTypeFloat8E4M3FNUZ = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ
	TensorElementDataTypeFloat8E5M2     = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2
	TensorElementDataTypeFloat8E5M2FNUZ = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ

	// Int4 types were introduced in ONNX 1.16. See
	// https://onnx.ai/onnx/technical/int4.html
	TensorElementDataTypeUint4 = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4
	TensorElementDataTypeInt4  = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4
)

func (t TensorElementDataType) String() string {
	switch t {
	case TensorElementDataTypeUndefined:
		return "ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED"
	case TensorElementDataTypeFloat:
		return "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT"
	case TensorElementDataTypeUint8:
		return "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8"
	case TensorElementDataTypeInt8:
		return "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8"
	case TensorElementDataTypeUint16:
		return "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16"
	case TensorElementDataTypeInt16:
		return "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16"
	case TensorElementDataTypeInt32:
		return "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32"
	case TensorElementDataTypeInt64:
		return "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64"
	case TensorElementDataTypeString:
		return "ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING"
	case TensorElementDataTypeBool:
		return "ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL"
	case TensorElementDataTypeFloat16:
		return "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16"
	case TensorElementDataTypeDouble:
		return "ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE"
	case TensorElementDataTypeUint32:
		return "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32"
	case TensorElementDataTypeUint64:
		return "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64"
	case TensorElementDataTypeComplex64:
		return "ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64"
	case TensorElementDataTypeComplex128:
		return "ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128"
	case TensorElementDataTypeBFloat16:
		return "ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16"
	case TensorElementDataTypeFloat8E4M3FN:
		return "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN"
	case TensorElementDataTypeFloat8E4M3FNUZ:
		return "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ"
	case TensorElementDataTypeFloat8E5M2:
		return "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2"
	case TensorElementDataTypeFloat8E5M2FNUZ:
		return "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ"
	case TensorElementDataTypeUint4:
		return "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4"
	case TensorElementDataTypeInt4:
		return "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4"
	}
	return fmt.Sprintf("Unknown tensor element data type: %d", int(t))
}

// This wraps an ONNX_TYPE_SEQUENCE OrtValue. Satisfies the Value interface,
// though Tensor-related functions such as ZeroContents() may be no-ops.
type Sequence struct {
	ortValue *C.OrtValue
	// We'll stash the values in the sequence here, so we don't need to look
	// them up, and so that users don't need to remember to free them.
	contents []Value
}

// Returns the value at the given index in the sequence or map. (In a map,
// index 0 is for keys, and 1 is for values.) Used internally when initializing
// a go Sequence or Map object.
func getSequenceOrMapValue(sequenceOrMap *C.OrtValue,
	index int64) (Value, error) {
	var result *C.OrtValue
	status := C.GetValue(sequenceOrMap, C.int(index), &result)
	if status != nil {
		return nil, fmt.Errorf("Error getting value of index %d: %s", index,
			statusToError(status))
	}
	return createGoValueFromOrtValue(result)
}

// Creates a new ONNX sequence with the given contents. The returned Sequence
// must be Destroyed by the caller when no longer needed. Destroying the
// Sequence created by this function does _not_ destroy the Values it was
// created with, so the caller is still responsible for destroying them
// as well.
//
// The contents of a sequence are subject to additional constraints. I can't
// find mention of some of these in the C API docs, but they are enforced by
// the onnxruntime API. Notably: all elements of the sequence must have the
// same type, and all elements must be either maps or tensors. Finally, the
// sequence must contain at least one element, and none of the elements may be
// nil. There may be other constraints that I am unaware of, as well.
func NewSequence(contents []Value) (*Sequence, error) {
	if !IsInitialized() {
		return nil, NotInitializedError
	}
	length := int64(len(contents))
	if length == 0 {
		return nil, fmt.Errorf("Sequences must contain at least 1 element")
	}
	ortValues := make([]*C.OrtValue, length)
	for i, v := range contents {
		if v == nil {
			return nil, fmt.Errorf("Sequences must not contain nil (index "+
				"%d was nil)", i)
		}
		ortValues[i] = v.GetInternals().ortValue
	}

	var sequence *C.OrtValue
	status := C.CreateOrtValue(&(ortValues[0]), C.size_t(length),
		C.ONNX_TYPE_SEQUENCE, &sequence)
	if status != nil {
		return nil, fmt.Errorf("Error creating ORT sequence: %s",
			statusToError(status))
	}

	// Finally, we want to get each OrtValue from the sequence itself, but we
	// already have a function to do this in the case of onnxruntime-allocated
	// sequences.
	toReturn, e := createSequenceFromOrtValue(sequence)
	if e != nil {
		// createSequenceFromOrtValue destroys the sequence on error.
		return nil, fmt.Errorf("Error creating go Sequence from sequence "+
			"OrtValue: %w", e)
	}
	return toReturn, nil
}

// Returns the list of values in the sequence. Each of these values should
// _not_ be Destroy()'ed by the caller, they will be automatically destroyed
// upon calling Destroy() on the sequence. If this sequence was created via
// NewSequence, these are not the same Values that the sequence was created
// with, though if they are tensors they should still refer to the same
// underlying data.
func (s *Sequence) GetValues() ([]Value, error) {
	return s.contents, nil
}

func (s *Sequence) Destroy() error {
	C.ReleaseOrtValue(s.ortValue)
	var e error
	for _, v := range s.contents {
		if v != nil {
			// Just return the last error if any of these returns an error.
			e2 := v.Destroy()
			if e2 != nil {
				e = e2
			}
		}
	}
	s.ortValue = nil
	s.contents = nil
	return e
}

// This returns a 1-dimensional Shape containing a single element: the number
// of elements the sequence. Typically, Sequence users should prefer calling
// len(s.GetValues()) over this function. This function only exists to maintain
// compatibility with the Value interface.
func (s *Sequence) GetShape() Shape {
	return NewShape(int64(len(s.contents)))
}

// Always returns ONNXTypeSequence
func (s *Sequence) GetONNXType() ONNXType {
	return ONNXTypeSequence
}

// This function is meaningless for a Sequence and shouldn't be used. The
// return value is always TENSOR_ELEMENT_DATA_TYPE_UNDEFINED for now, but this
// may change in the future. This function is only present for compatibility
// with the Value interface and should not be relied on for sequences.
func (s *Sequence) DataType() C.ONNXTensorElementDataType {
	return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
}

// This function does nothing for a Sequence, and is only present for
// compatibility with the Value interface.
func (s *Sequence) ZeroContents() {
}

func (s *Sequence) GetInternals() *ValueInternalData {
	return &ValueInternalData{
		ortValue: s.ortValue,
	}
}

// This wraps an ONNX_TYPE_MAP OrtValue. Satisfies the Value interface,
// though Tensor-related functions such as ZeroContents() may be no-ops.
type Map struct {
	ortValue *C.OrtValue
	// An onnxruntime map is really just two tensors, keys and values, that
	// must be the same length. These Values will be cleaned up when calling
	// Map.Destroy.
	keys   Value
	values Value
}

// Creates a new ONNX map that maps the given keys tensor to the given values
// tensor. Destroying the Map created by this function does _not_ destroy these
// keys and values tensors; the caller is still responsible for destroying
// them.
//
// Internally, creating a Map requires two tensors of the same length, and
// with constraints on type.  For example, keys are not allowed to be floats
// (at least currently). (At the time of writing, this has only been confirmed
// to work with int64 keys.) There may be many other constraints enforced by
// the underlying C API.
func NewMap(keys, values Value) (*Map, error) {
	if !IsInitialized() {
		return nil, NotInitializedError
	}

	newMapArgs := []*C.OrtValue{
		keys.GetInternals().ortValue,
		values.GetInternals().ortValue,
	}
	var result *C.OrtValue
	status := C.CreateOrtValue(&(newMapArgs[0]), 2, C.ONNX_TYPE_MAP, &result)
	if status != nil {
		return nil, fmt.Errorf("Error creating ORT map: %s",
			statusToError(status))
	}

	// We need to obtain internal references to the keys and values allocated
	// by onnxruntime. createMapFromOrtValue does this for us.
	toReturn, e := createMapFromOrtValue(result)
	if e != nil {
		// createMapFromOrtValue already destroys the OrtValue on error.
		return nil, fmt.Errorf("Error creating Map instance from map "+
			"OrtValue: %w", e)
	}
	return toReturn, nil
}

// Wraps the creation of an ONNX map from a Go map. K is the key type, and V is
// the value type. Be aware that constraints on these types exist based on
// what ONNX supports. See the comment on NewMap.
func NewMapFromGoMap[K, V TensorData](m map[K]V) (*Map, error) {
	if !IsInitialized() {
		return nil, NotInitializedError
	}
	keysSlice := make([]K, len(m))
	valuesSlice := make([]V, len(m))
	i := 0
	for k, v := range m {
		keysSlice[i] = k
		valuesSlice[i] = v
		i++
	}
	tensorShape := NewShape(int64(len(m)))
	keysTensor, e := NewTensor(tensorShape, keysSlice)
	if e != nil {
		return nil, fmt.Errorf("Error creating keys tensor for map: %w", e)
	}
	defer keysTensor.Destroy()
	valuesTensor, e := NewTensor(tensorShape, valuesSlice)
	if e != nil {
		return nil, fmt.Errorf("Error creating values tensor for map: %w", e)
	}
	defer valuesTensor.Destroy()
	toReturn, e := NewMap(keysTensor, valuesTensor)
	if e != nil {
		return nil, fmt.Errorf("Error creating map from key and value "+
			"tensors: %w", e)
	}
	return toReturn, nil
}

// Returns two Tensors containing the keys and values, respectively. These
// tensors should _not_ be Destroyed by users; they will be automatically
// cleaned up when m.Destroy() is called. These are _not_ the same Value
// instances that were passed to NewMap, and these should not be modified by
// users.
func (m *Map) GetKeysAndValues() (Value, Value, error) {
	return m.keys, m.values, nil
}

func (m *Map) Destroy() error {
	C.ReleaseOrtValue(m.ortValue)
	// Just return the last error if either of these returns an error.
	var e error
	e2 := m.keys.Destroy()
	if e2 != nil {
		e = e2
	}
	e2 = m.values.Destroy()
	if e2 != nil {
		e = e2
	}
	m.ortValue = nil
	m.keys = nil
	m.values = nil
	return e
}

// Always returns ONNXTypeMap
func (m *Map) GetONNXType() ONNXType {
	return ONNXTypeMap
}

// Returns the shape of the map's keys Tensor. Essentially, this can be used
// to determine the number of key/value pairs in the map.
func (m *Map) GetShape() Shape {
	return m.keys.GetShape()
}

func (m *Map) GetInternals() *ValueInternalData {
	return &ValueInternalData{
		ortValue: m.ortValue,
	}
}

// As with Sequence.ZeroContents(), this is a no-op (at least for now), and is
// only present for compatibility with the Value interface.
func (m *Map) ZeroContents() {
}

// As with a Sequence, this always returns the undefined data type and is only
// present for compatibility with the Value interface.
func (m *Map) DataType() C.ONNXTensorElementDataType {
	return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
}

// Wraps the ONNXType enum in C.
type ONNXType int

const (
	ONNXTypeUnknown      = C.ONNX_TYPE_UNKNOWN
	ONNXTypeTensor       = C.ONNX_TYPE_TENSOR
	ONNXTypeSequence     = C.ONNX_TYPE_SEQUENCE
	ONNXTypeMap          = C.ONNX_TYPE_MAP
	ONNXTypeOpaque       = C.ONNX_TYPE_OPAQUE
	ONNXTypeSparseTensor = C.ONNX_TYPE_SPARSETENSOR
	ONNXTypeOptional     = C.ONNX_TYPE_OPTIONAL
)

func (t ONNXType) String() string {
	switch t {
	case ONNXTypeUnknown:
		return "ONNX_TYPE_UNKNOWN"
	case ONNXTypeTensor:
		return "ONNX_TYPE_TENSOR"
	case ONNXTypeSequence:
		return "ONNX_TYPE_SEQUENCE"
	case ONNXTypeMap:
		return "ONNX_TYPE_MAP"
	case ONNXTypeOpaque:
		return "ONNX_TYPE_OPAQUE"
	case ONNXTypeSparseTensor:
		return "ONNX_TYPE_SPARSE_TENSOR"
	case ONNXTypeOptional:
		return "ONNX_TYPE_OPTIONAL"
	}
	return fmt.Sprintf("Unknown ONNX type: %d", int(t))
}

// This satisfies the Value interface, but is intended to allow users to
// provide tensors of types that may not be supported by the generic typed
// Tensor[T] struct. Instead, CustomDataTensors are backed by a slice of bytes,
// using a user-provided shape and type from the ONNXTensorElementDataType
// enum.
type CustomDataTensor struct {
	data     []byte
	dataType C.ONNXTensorElementDataType
	shape    Shape
	ortValue *C.OrtValue
}

// Creates and returns a new CustomDataTensor using the given bytes as the
// underlying data slice. Apart from ensuring that the provided data slice is
// non-empty, this function mostly delegates validation of the provided data to
// the C onnxruntime library. For example, it is the caller's responsibility to
// ensure that the provided dataType and data slice are valid and correctly
// sized for the specified shape. If this returns successfully, the caller must
// call the returned tensor's Destroy() function to free it when no longer in
// use.
func NewCustomDataTensor(s Shape, data []byte,
	dataType TensorElementDataType) (*CustomDataTensor, error) {
	if !IsInitialized() {
		return nil, NotInitializedError
	}
	e := s.Validate()
	if e != nil {
		return nil, fmt.Errorf("Invalid tensor shape: %w", e)
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("A CustomDataTensor requires at least one " +
			"byte of data")
	}
	dt := C.ONNXTensorElementDataType(dataType)
	var ortValue *C.OrtValue

	status := C.CreateOrtTensorWithShape(unsafe.Pointer(&data[0]),
		C.size_t(len(data)), (*C.int64_t)(unsafe.Pointer(&s[0])),
		C.int64_t(len(s)), ortMemoryInfo, dt, &ortValue)
	if status != nil {
		return nil, fmt.Errorf("ORT API error creating tensor: %s",
			statusToError(status))
	}
	toReturn := CustomDataTensor{
		data:     data,
		dataType: dt,
		shape:    s.Clone(),
		ortValue: ortValue,
	}
	return &toReturn, nil
}

func (t *CustomDataTensor) Destroy() error {
	C.ReleaseOrtValue(t.ortValue)
	t.ortValue = nil
	t.data = nil
	t.shape = nil
	t.dataType = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
	return nil
}

func (t *CustomDataTensor) DataType() C.ONNXTensorElementDataType {
	return t.dataType
}

func (t *CustomDataTensor) GetShape() Shape {
	return t.shape.Clone()
}

func (t *CustomDataTensor) GetInternals() *ValueInternalData {
	return &ValueInternalData{
		ortValue: t.ortValue,
	}
}

// Always returns ONNXTypeTensor, even if the CustomDataTensor is invalid for
// some reason.
func (t *CustomDataTensor) GetONNXType() ONNXType {
	return ONNXTypeTensor
}

// Sets all bytes in the data slice to 0.
func (t *CustomDataTensor) ZeroContents() {
	C.memset(unsafe.Pointer(&t.data[0]), 0, C.size_t(len(t.data)))
}

// Returns the same slice that was passed to NewCustomDataTensor.
func (t *CustomDataTensor) GetData() []byte {
	return t.data
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

// Enable/Disable the usage of the memory arena on CPU.
// Arena may pre-allocate memory for future usage.
func (o *SessionOptions) SetCpuMemArena(isEnabled bool) error {
	n := 0
	if isEnabled {
		n = 1
	}
	status := C.SetCpuMemArena(o.o, C.int(n))
	if status != nil {
		return statusToError(status)
	}
	return nil
}

// Enable/Disable the memory pattern optimization.
// If this is enabled memory is preallocated if all shapes are known.
func (o *SessionOptions) SetMemPattern(isEnabled bool) error {
	n := 0
	if isEnabled {
		n = 1
	}
	status := C.SetMemPattern(o.o, C.int(n))
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

// Enables the DirectML backend for the given session options on supported
// platforms. See the notes on device_id in coreml_provider_factory.h in the
// onnxruntime source code, but a device ID of 0 should correspond to the
// default device, "which is typically the primary display GPU" according to
// the docs.
func (o *SessionOptions) AppendExecutionProviderDirectML(deviceID int) error {
	status := C.AppendExecutionProviderDirectML(o.o, C.int(deviceID))
	if status != nil {
		return statusToError(status)
	}
	return nil
}

// Enables the OpenVINO backend for the given session options on supported
// platforms. See
// https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#summary-of-options
// for a list of supported keys and values that can be passed in the options
// map.
func (o *SessionOptions) AppendExecutionProviderOpenVINO(
	options map[string]string) error {
	// There's probably a more concise way to do this, but we don't want to
	// do "&(keys[0])" if keys is an empty slice, so we'll declare the null
	// ptrs ahead of time and only set them if we know the slices aren't empty.
	var keysPtr, valuesPtr **C.char
	if len(options) != 0 {
		keys, values := mapToCStrings(options)
		defer freeCStrings(keys)
		defer freeCStrings(values)
		keysPtr = &(keys[0])
		valuesPtr = &(values[0])
	}

	status := C.AppendExecutionProviderOpenVINOV2(o.o, keysPtr, valuesPtr,
		C.int(len(options)))
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

// A wrapper around the OrtModelMetadata C struct. Must be freed by calling
// Destroy() on it when it's no longer needed.
type ModelMetadata struct {
	m *C.OrtModelMetadata
}

// Frees internal state required by the model metadata. Users are responsible
// for calling this on any ModelMetadata instance after it's no longer needed.
func (m *ModelMetadata) Destroy() error {
	if m.m != nil {
		C.ReleaseModelMetadata(m.m)
		m.m = nil
	}
	return nil
}

// Takes a C string allocated using the default ORT allocator, converts it to
// a Go string, and frees the C copy. Returns an error if one occurs. Returns
// an empty string with no error if s is nil. Obviously, s is invalid after
// this returns.
func convertORTString(s *C.char) (string, error) {
	if s == nil {
		return "", nil
	}
	// Unfortunately, onnxruntime wants to use custom allocators to allocate
	// data such as strings, which are rather obtuse to customize. Therefore,
	// our C code always specifies the default ORT allocator when possible. We
	// move any strings ORT allocates into Go strings so we can free the C
	// versions as soon as possible.
	toReturn := C.GoString(s)
	status := C.FreeWithDefaultORTAllocator(unsafe.Pointer(s))
	if status != nil {
		return toReturn, statusToError(status)
	}
	return toReturn, nil
}

// Returns the producer name associated with the model metadata, or an error if
// the name can't be obtained.
func (m *ModelMetadata) GetProducerName() (string, error) {
	var cName *C.char
	status := C.ModelMetadataGetProducerName(m.m, &cName)
	if status != nil {
		return "", statusToError(status)
	}
	return convertORTString(cName)
}

// Returns the graph name associated with the model metadata, or an error if
// the name can't be obtained.
func (m *ModelMetadata) GetGraphName() (string, error) {
	var cName *C.char
	status := C.ModelMetadataGetGraphName(m.m, &cName)
	if status != nil {
		return "", statusToError(status)
	}
	return convertORTString(cName)
}

// Returns the domain associated with the model metadata, or an error if the
// domain can't be obtained.
func (m *ModelMetadata) GetDomain() (string, error) {
	var cDomain *C.char
	status := C.ModelMetadataGetDomain(m.m, &cDomain)
	if status != nil {
		return "", statusToError(status)
	}
	return convertORTString(cDomain)
}

// Returns the description associated with the model metadata, or an error if
// the description can't be obtained.
func (m *ModelMetadata) GetDescription() (string, error) {
	var cDescription *C.char
	status := C.ModelMetadataGetDescription(m.m, &cDescription)
	if status != nil {
		return "", statusToError(status)
	}
	return convertORTString(cDescription)
}

// Returns the version number in the model metadata, or an error if one occurs.
func (m *ModelMetadata) GetVersion() (int64, error) {
	var version C.int64_t
	status := C.ModelMetadataGetVersion(m.m, &version)
	if status != nil {
		return 0, statusToError(status)
	}
	return int64(version), nil
}

// Looks up and returns the string associated with the given key in the custom
// metadata map. Returns a blank string and 'false' if the key isn't in the
// map. (A key that's in the map but set to a blank string will
// return "" and true instead.)
//
// NOTE: It is unclear from the onnxruntime documentation for this function
// whether an error will be returned if the key isn't present. At the time of
// writing (1.17.1) the docs only state that no value is returned, not whether
// an error occurs.
func (m *ModelMetadata) LookupCustomMetadataMap(key string) (string, bool, error) {
	var cValue *C.char
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	status := C.ModelMetadataLookupCustomMetadataMap(m.m, cKey, &cValue)
	if status != nil {
		return "", false, statusToError(status)
	}
	if cValue == nil {
		return "", false, nil
	}
	value, e := convertORTString(cValue)
	return value, true, e
}

// Returns a list of keys that are present in the custom metadata map. Returns
// an empty slice or nil if no keys are in the map.
//
// NOTE: It is unclear from the docs whether an empty custom metadata map will
// cause the underlying C function to return an error along with a NULL list,
// or whether it will only return a NULL list with no error.
func (m *ModelMetadata) GetCustomMetadataMapKeys() ([]string, error) {
	var keyCount C.int64_t
	var cKeys **C.char
	status := C.ModelMetadataGetCustomMetadataMapKeys(m.m, &cKeys, &keyCount)
	if status != nil {
		return nil, statusToError(status)
	}
	if cKeys == nil {
		// We got no keys in the map and no error return
		return nil, nil
	}
	if keyCount == 0 {
		// We have a non-NULL but empty list of C pointers, so we'll still
		// free it here.
		status := C.FreeWithDefaultORTAllocator(unsafe.Pointer(cKeys))
		if status != nil {
			return nil, statusToError(status)
		}
		return nil, nil
	}

	// The slice allows us to index into the array of C-string pointers.
	cKeySlice := unsafe.Slice(cKeys, int64(keyCount))
	toReturn := make([]string, len(cKeySlice))
	var e error
	for i, s := range cKeySlice {
		// We won't check for errors until after the loop, because we want to
		// continue trying to free all of the strings regardless of whether an
		// error occurs for one of them.
		toReturn[i], e = convertORTString(s)
		cKeySlice[i] = nil
	}
	// At this point, we've done our best to convert and free all of the ORT-
	// allocated C strings, but we still need to free the array itself, which
	// we attempt regardless of whether an error occurred during the string
	// processing.
	status = C.FreeWithDefaultORTAllocator(unsafe.Pointer(cKeys))
	cKeySlice = nil
	cKeys = nil
	if e != nil {
		return nil, fmt.Errorf("Error copying one or more C strings to Go: %w",
			e)
	}
	if status != nil {
		return nil, fmt.Errorf("Error freeing array of C strings: %w",
			statusToError(status))
	}

	return toReturn, nil
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

// Basically identical to createCSession, except uses a file path rather than
// a buffer of .onnx content.
func createCSessionFromFile(path string,
	options *SessionOptions) (*C.OrtSession, error) {
	if !IsInitialized() {
		return nil, NotInitializedError
	}
	cPath, e := createOrtCharString(path)
	if e != nil {
		return nil, fmt.Errorf("Unable to convert path to C path: %w", e)
	}
	var ortSession *C.OrtSession
	var ortSessionOptions *C.OrtSessionOptions
	if options != nil {
		ortSessionOptions = options.o
	}
	status := C.CreateSessionFromFile(cPath, ortEnv, &ortSession,
		ortSessionOptions)
	if status != nil {
		return nil, statusToError(status)
	}
	return ortSession, nil
}

// Initializes an AdvancedSession object without creating the session;
// essentially converting input and output names. Set the dynamicInputs
// argument to true if this will be used for a DynamicAdvancedSession; it will
// skip checks on the inputs and outputs []Values.
func newAdvancedSessionInternal(inputNames, outputNames []string,
	inputs, outputs []Value, dynamicInputs bool) (*AdvancedSession, error) {
	if !IsInitialized() {
		return nil, NotInitializedError
	}
	if !dynamicInputs {
		if len(inputs) == 0 {
			return nil, fmt.Errorf("No inputs were provided")
		}
		if len(outputs) == 0 {
			return nil, fmt.Errorf("No outputs were provided")
		}
		if len(inputs) != len(inputNames) {
			return nil, fmt.Errorf("Got %d inputs, but %d input names",
				len(inputs), len(inputNames))
		}
		if len(outputs) != len(outputNames) {
			return nil, fmt.Errorf("Got %d outputs, but %d output names",
				len(outputs), len(outputNames))
		}
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
	var inputOrtValues, outputOrtValues []*C.OrtValue
	if !dynamicInputs {
		inputOrtValues = make([]*C.OrtValue, len(inputs))
		outputOrtValues = make([]*C.OrtValue, len(outputs))
		for i, v := range inputs {
			inputOrtValues[i] = v.GetInternals().ortValue
		}
		for i, v := range outputs {
			outputOrtValues[i] = v.GetInternals().ortValue
		}
	}
	return &AdvancedSession{
		ortSession:  nil,
		inputNames:  cInputNames,
		outputNames: cOutputNames,
		inputs:      inputOrtValues,
		outputs:     outputOrtValues,
	}, nil
}

// The same as NewAdvancedSession, but takes a slice of bytes containing the
// .onnx network rather than a file path.
func NewAdvancedSessionWithONNXData(onnxData []byte, inputNames,
	outputNames []string, inputs, outputs []Value,
	options *SessionOptions) (*AdvancedSession, error) {
	toReturn, e := newAdvancedSessionInternal(inputNames, outputNames, inputs,
		outputs, false)
	if e != nil {
		return nil, e
	}
	toReturn.ortSession, e = createCSession(onnxData, options)
	if e != nil {
		toReturn.Destroy()
		return nil, fmt.Errorf("Error creating C session: %w", e)
	}
	return toReturn, nil
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
	inputs, outputs []Value,
	options *SessionOptions) (*AdvancedSession, error) {
	toReturn, e := newAdvancedSessionInternal(inputNames, outputNames, inputs,
		outputs, false)
	if e != nil {
		return nil, e
	}
	toReturn.ortSession, e = createCSessionFromFile(onnxFilePath, options)
	if e != nil {
		toReturn.Destroy()
		return nil, fmt.Errorf("Error creating C session from file: %w", e)
	}
	return toReturn, nil
}

func (s *AdvancedSession) Destroy() error {
	// Including the check that ortSession is not nil allows the Destroy()
	// function to be used on AdvancedSessions that are partially initialized,
	// which we do internally.
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

// Creates and returns a ModelMetadata instance for this session's model. The
// returned metadata must be freed using its Destroy() function when no longer
// needed.
func (s *AdvancedSession) GetModelMetadata() (*ModelMetadata, error) {
	var m *C.OrtModelMetadata
	status := C.SessionGetModelMetadata(s.ortSession, &m)
	if status != nil {
		return nil, statusToError(status)
	}
	return &ModelMetadata{
		m: m,
	}, nil
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
	s, e := newAdvancedSessionInternal(inputNames, outputNames, nil, nil, true)
	if e != nil {
		return nil, fmt.Errorf("Error creating internal AdvancedSession: %w",
			e)
	}
	s.ortSession, e = createCSession(onnxData, options)
	if e != nil {
		s.Destroy()
		return nil, fmt.Errorf("Error creating C session: %w", e)
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
	s, e := newAdvancedSessionInternal(inputNames, outputNames, nil, nil, true)
	if e != nil {
		return nil, fmt.Errorf("Error creating internal AdvancedSession: %w",
			e)
	}
	s.ortSession, e = createCSessionFromFile(onnxFilePath, options)
	if e != nil {
		s.Destroy()
		return nil, fmt.Errorf("Error creating C session from file: %w", e)
	}
	return &DynamicAdvancedSession{
		s: s,
	}, nil
}

func (s *DynamicAdvancedSession) Destroy() error {
	return s.s.Destroy()
}

func createTensorWithCData[T TensorData](shape Shape, data unsafe.Pointer) (*Tensor[T], error) {
	totalSize := shape.FlattenedSize()
	actualData := unsafe.Slice((*T)(data), totalSize)
	dataCopy := make([]T, totalSize)
	copy(dataCopy, actualData)
	return NewTensor[T](shape, dataCopy)
}

// Returns the Shape described by a TensorTypeAndShapeInfo instance.
func getShapeFromInfo(t *C.OrtTensorTypeAndShapeInfo) (Shape, error) {
	var dimCount C.size_t
	status := C.GetDimensionsCount(t, &dimCount)
	if status != nil {
		return nil, fmt.Errorf("Error getting dimension count: %w",
			statusToError(status))
	}
	shape := make(Shape, dimCount)
	status = C.GetDimensions(t, (*C.int64_t)(&shape[0]), dimCount)
	if status != nil {
		return nil, fmt.Errorf("Error getting shape dimensions: %w",
			statusToError(status))
	}
	return shape, nil
}

// Returns the ONNXType associated with a C OrtValue.
func getValueType(v *C.OrtValue) (ONNXType, error) {
	var t C.enum_ONNXType
	status := C.GetValueType(v, &t)
	if status != nil {
		return ONNXTypeUnknown, fmt.Errorf("Error looking up type for "+
			"OrtValue: %s", statusToError(status))
	}
	return ONNXType(t), nil
}

// Returns the "count" associated with an OrtValue. Mostly useful for
// sequences. Should always return 2 for a map. Not sure what it returns for
// Tensors, but that shouldn't matter.
func getValueCount(v *C.OrtValue) (int64, error) {
	var size C.size_t
	status := C.GetValueCount(v, &size)
	if status != nil {
		return 0, fmt.Errorf("Error getting non tensor count for OrtValue: %s",
			statusToError(status))
	}
	return int64(size), nil
}

// Takes an OrtValue and returns an appropriate Go value wrapping it, or at
// least an equivalent go value in case v is a Tensor. The Value v should
// _not_ be released after calling this function; it will either be released
// internally or released when the returned Value is Destroy()'d. (Callers must
// destroy the returned value.)
//
// If this function fails, v will be released.
func createGoValueFromOrtValue(v *C.OrtValue) (Value, error) {
	if v == nil {
		return nil, fmt.Errorf("Internal error: got nil argument to " +
			"createGoValueFromOrtValue")
	}
	valueType, e := getValueType(v)
	if e != nil {
		C.ReleaseOrtValue(v)
		return nil, e
	}
	switch valueType {
	case ONNXTypeTensor:
		return createTensorFromOrtValue(v)
	case ONNXTypeSequence:
		return createSequenceFromOrtValue(v)
	case ONNXTypeMap:
		return createMapFromOrtValue(v)
	default:
		break
	}
	C.ReleaseOrtValue(v)
	return nil, fmt.Errorf("It is currently not supported to create a Go "+
		"value from OrtValues with ONNXType = %s", valueType)
}

// Must only be called if v is known to be of type ONNXTensor. Returns a Tensor
// wrapping v with the correct Go type. This function always copies v's
// contents into a new Tensor backed by a Go-managed slice and releases v.
func createTensorFromOrtValue(v *C.OrtValue) (Value, error) {
	// Either in the case of error or otherwise, we'll release v. The issue is
	// that GetTensorMutableData() becomes invalid after v is Released, so we
	// can't release v if a reference to the slice returned by GetData is
	// still referred to outside of the tensor. We work around this by copying
	// the data into a new tensor and releasing the original.
	defer C.ReleaseOrtValue(v)

	var pInfo *C.OrtTensorTypeAndShapeInfo
	status := C.GetTensorTypeAndShape(v, &pInfo)
	if status != nil {
		return nil, fmt.Errorf("Error getting type and shape: %w",
			statusToError(status))
	}
	shape, e := getShapeFromInfo(pInfo)
	if e != nil {
		return nil, fmt.Errorf("Error getting shape from TypeAndShapeInfo: %w",
			e)
	}
	var tensorElementType C.ONNXTensorElementDataType
	status = C.GetTensorElementType(pInfo, (*uint32)(&tensorElementType))
	if status != nil {
		return nil, fmt.Errorf("Error getting tensor element type: %w",
			statusToError(status))
	}
	C.ReleaseTensorTypeAndShapeInfo(pInfo)
	var tensorData unsafe.Pointer
	status = C.GetTensorMutableData(v, &tensorData)
	if status != nil {
		return nil, fmt.Errorf("Error getting tensor mutable data: %w",
			statusToError(status))
	}

	switch tensorType := TensorElementDataType(tensorElementType); tensorType {
	case TensorElementDataTypeFloat:
		return createTensorWithCData[float32](shape, tensorData)
	case TensorElementDataTypeUint8:
		return createTensorWithCData[uint8](shape, tensorData)
	case TensorElementDataTypeInt8:
		return createTensorWithCData[int8](shape, tensorData)
	case TensorElementDataTypeUint16:
		return createTensorWithCData[uint16](shape, tensorData)
	case TensorElementDataTypeInt16:
		return createTensorWithCData[int16](shape, tensorData)
	case TensorElementDataTypeInt32:
		return createTensorWithCData[int32](shape, tensorData)
	case TensorElementDataTypeInt64:
		return createTensorWithCData[int64](shape, tensorData)
	case TensorElementDataTypeDouble:
		return createTensorWithCData[float64](shape, tensorData)
	case TensorElementDataTypeUint32:
		return createTensorWithCData[uint32](shape, tensorData)
	case TensorElementDataTypeUint64:
		return createTensorWithCData[uint64](shape, tensorData)
	default:
		totalSize := shape.FlattenedSize()
		actualData := unsafe.Slice((*byte)(tensorData), totalSize)
		dataCopy := make([]byte, totalSize)
		copy(dataCopy, actualData)
		return NewCustomDataTensor(shape, dataCopy, tensorType)
	}
}

// Must only be called if v is already known to be an ONNXTypeSequence. Returns
// a Sequence go type wrapping v. Releases v if an error occurs; otherwise v
// will be released when the returned Sequence is destroyed.
func createSequenceFromOrtValue(v *C.OrtValue) (*Sequence, error) {
	length, e := getValueCount(v)
	if e != nil {
		C.ReleaseOrtValue(v)
		return nil, fmt.Errorf("Error determining sequence length: %w", e)
	}

	// Retrieve all of the sequence's contents as Go values, too.
	internalValues := make([]Value, length)
	for i := range internalValues {
		internalValues[i], e = getSequenceOrMapValue(v, int64(i))
		if e != nil {
			// Clean up whatever values we already created.
			for j := 0; j < i; j++ {
				internalValues[i].Destroy()
			}
			C.ReleaseOrtValue(v)
			return nil, fmt.Errorf("Error retrieving sequence contents at "+
				"index %d: %w", i, e)
		}
	}

	return &Sequence{
		ortValue: v,
		contents: internalValues,
	}, nil
}

// Must only be called if v is already known to be an ONNXTypeMap. Returns a
// Map go type wrapping v. Releases v if an error occurs, otherwise v will be
// released when the returned Map is destroyed.
func createMapFromOrtValue(v *C.OrtValue) (*Map, error) {
	// Obtain the keys and values as tensors from the Map instance.
	keys, e := getSequenceOrMapValue(v, 0)
	if e != nil {
		C.ReleaseOrtValue(v)
		return nil, fmt.Errorf("Error getting keys tensor from map: %w", e)
	}
	values, e := getSequenceOrMapValue(v, 1)
	if e != nil {
		keys.Destroy()
		C.ReleaseOrtValue(v)
		return nil, fmt.Errorf("Error getting values tensor from map: %w", e)
	}

	return &Map{
		ortValue: v,
		keys:     keys,
		values:   values,
	}, nil
}

// Runs the network on the given input and output tensors. The number of input
// and output tensors must match the number (and order) of the input and output
// names specified to NewDynamicAdvancedSession. If a given output is nil, it
// will be allocated and the slice will be modified to include the new Value.
// Any new Value allocated in this way must be freed by calling Destroy on it.
func (s *DynamicAdvancedSession) Run(inputs, outputs []Value) error {
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
		if v == nil {
			// Leave any output that needs to be allocated as nil.
			continue
		}
		outputValues[i] = v.GetInternals().ortValue
	}

	status := C.RunOrtSession(s.s.ortSession, &inputValues[0],
		&s.s.inputNames[0], C.int(len(inputs)), &outputValues[0],
		&s.s.outputNames[0], C.int(len(outputs)))
	if status != nil {
		return fmt.Errorf("Error running network: %w", statusToError(status))
	}
	// Convert any automatically-allocated output to a go Value.
	for i, v := range outputs {
		if v != nil {
			continue
		}
		var err error
		outputs[i], err = createGoValueFromOrtValue(outputValues[i])
		if err != nil {
			return fmt.Errorf("Error creating tensor from ort: %w", err)
		}
	}
	return nil
}

// Creates and returns a ModelMetadata instance for this session's model. The
// returned metadata must be freed using its Destroy() function when no longer
// needed.
func (s *DynamicAdvancedSession) GetModelMetadata() (*ModelMetadata, error) {
	return s.s.GetModelMetadata()
}

// Holds information about the name, shape, and type of an input or output to a
// ONNX network.
type InputOutputInfo struct {
	// The name of the input or output
	Name string
	// The higher-level "type" of the output; whether it's a tensor, sequence,
	// map, etc.
	OrtValueType ONNXType
	// The input or output's dimensions, if it's a tensor. This should be
	// ignored for non-tensor types.
	Dimensions Shape
	// The type of element in the input or output, if it's a tensor. This
	// should be ignored for non-tensor types.
	DataType TensorElementDataType
}

func (n *InputOutputInfo) String() string {
	switch n.OrtValueType {
	case ONNXTypeUnknown:
		return fmt.Sprintf("Unknown ONNX type: %s", n.Name)
	case ONNXTypeTensor:
		return fmt.Sprintf("Tensor \"%s\": %s, %s", n.Name, n.Dimensions,
			n.DataType)
	case ONNXTypeSequence:
		return fmt.Sprintf("Sequence \"%s\"", n.Name)
	case ONNXTypeMap:
		return fmt.Sprintf("Map \"%s\"", n.Name)
	case ONNXTypeOpaque:
		return fmt.Sprintf("Opaque \"%s\"", n.Name)
	case ONNXTypeSparseTensor:
		return fmt.Sprintf("Sparse tensor \"%s\": dense shape %s, %s",
			n.Name, n.Dimensions, n.DataType)
	case ONNXTypeOptional:
		return fmt.Sprintf("Optional \"%s\"", n.Name)
	default:
		break
	}
	// We'll use the ONNXType String() output if we don't know the type.
	return fmt.Sprintf("%s: \"%s\"", n.OrtValueType, n.Name)
}

// Sets o.OrtValueType, o.DataType, and o.Dimensions from the contents of t.
func (o *InputOutputInfo) fillFromTypeInfo(t *C.OrtTypeInfo) error {
	var onnxType C.enum_ONNXType
	status := C.GetONNXTypeFromTypeInfo(t, &onnxType)
	if status != nil {
		return fmt.Errorf("Error getting ONNX type: %s", statusToError(status))
	}
	o.OrtValueType = ONNXType(onnxType)
	o.Dimensions = nil
	o.DataType = TensorElementDataTypeUndefined

	// We only fill in element type and dimensions if we're dealing with a
	// tensor of some sort.
	isTensorType := (o.OrtValueType == ONNXTypeTensor) ||
		(o.OrtValueType == ONNXTypeSparseTensor)
	if !isTensorType {
		return nil
	}

	// OrtTensorTypeAndShapeInfo pointers should *not* be released if they're
	// obtained via CastTypeInfoToTensorInfo.
	var typeAndShapeInfo *C.OrtTensorTypeAndShapeInfo
	status = C.CastTypeInfoToTensorInfo(t, &typeAndShapeInfo)
	if status != nil {
		return fmt.Errorf("Error getting type and shape info: %w",
			statusToError(status))
	}
	if typeAndShapeInfo == nil {
		return fmt.Errorf("Didn't get type and shape info for an OrtTypeInfo" +
			"(it may not be a tensor type?)")
	}
	var e error
	o.Dimensions, e = getShapeFromInfo(typeAndShapeInfo)
	if e != nil {
		return fmt.Errorf("Error getting shape from typeAndShapeInfo: %w", e)
	}
	var tensorElementType C.ONNXTensorElementDataType
	status = C.GetTensorElementType(typeAndShapeInfo,
		(*uint32)(&tensorElementType))
	if status != nil {
		return fmt.Errorf("Error getting data type from typeAndShapeInfo: %w",
			statusToError(status))
	}
	o.DataType = TensorElementDataType(tensorElementType)
	return nil
}

// Fills dst with information about the session's i'th input.
func getSessionInputInfo(s *C.OrtSession, i int, dst *InputOutputInfo) error {
	var cName *C.char
	var e error
	status := C.SessionGetInputName(s, C.size_t(i), &cName)
	if status != nil {
		return fmt.Errorf("Error getting name: %w", statusToError(status))
	}
	dst.Name, e = convertORTString(cName)
	if e != nil {
		return fmt.Errorf("Error converting C name to Go string: %w", e)
	}

	// Session inputs are reported as OrtTypeInfo structs, though usually we
	// want a tensor-specific OrtTensorTypeAndShapeInfo struct, which we can
	// get from the type info.
	var typeInfo *C.OrtTypeInfo
	status = C.SessionGetInputTypeInfo(s, C.size_t(i), &typeInfo)
	if status != nil {
		return fmt.Errorf("Error getting type info: %w", statusToError(status))
	}
	defer C.ReleaseTypeInfo(typeInfo)
	e = dst.fillFromTypeInfo(typeInfo)
	if e != nil {
		return e
	}
	return nil
}

// Fills dst with information about the session's i'th output.
func getSessionOutputInfo(s *C.OrtSession, i int, dst *InputOutputInfo) error {
	// This is basically identical to getSessionInputInfo.
	var cName *C.char
	var e error
	status := C.SessionGetOutputName(s, C.size_t(i), &cName)
	if status != nil {
		return fmt.Errorf("Error getting name: %w", statusToError(status))
	}
	dst.Name, e = convertORTString(cName)
	if e != nil {
		return fmt.Errorf("Error converting C name to Go string: %w", e)
	}
	var typeInfo *C.OrtTypeInfo
	status = C.SessionGetOutputTypeInfo(s, C.size_t(i), &typeInfo)
	if status != nil {
		return fmt.Errorf("Error getting type info: %w", statusToError(status))
	}
	defer C.ReleaseTypeInfo(typeInfo)
	e = dst.fillFromTypeInfo(typeInfo)
	if e != nil {
		return e
	}
	return nil
}

// Takes an initialized OrtSession and returns slices of info for each input
// and output, respectively. Used internally by GetInputOutputInfo, etc.
func getInputOutputInfoFromCSession(s *C.OrtSession) ([]InputOutputInfo,
	[]InputOutputInfo, error) {
	var e error

	// Allocate the structs to hold the results.
	var inputCount, outputCount C.size_t
	status := C.SessionGetInputCount(s, &inputCount)
	if status != nil {
		return nil, nil, statusToError(status)
	}
	inputs := make([]InputOutputInfo, inputCount)
	status = C.SessionGetOutputCount(s, &outputCount)
	if status != nil {
		return nil, nil, statusToError(status)
	}
	outputs := make([]InputOutputInfo, outputCount)

	// Get the results for each input and output.
	for i := 0; i < int(inputCount); i++ {
		e = getSessionInputInfo(s, i, &(inputs[i]))
		if e != nil {
			return nil, nil, fmt.Errorf("Error getting information about "+
				"input %d: %w", i, e)
		}
	}
	for i := 0; i < int(outputCount); i++ {
		e = getSessionOutputInfo(s, i, &(outputs[i]))
		if e != nil {
			return nil, nil, fmt.Errorf("Error getting information about "+
				"output %d: %w", i, e)
		}
	}
	return inputs, outputs, nil
}

// Takes a path to a .onnx file, and returns a list of inputs and a list of
// outputs, respectively. Will open, read, and close the .onnx file to get the
// information. InitializeEnvironment() must have been called prior to using
// this function. Warning: this function requires loading the .onnx file into a
// temporary onnxruntime session, which may be an expensive operation.
//
// For now, this may fail if the network has any non-tensor inputs or inputs
// that don't have a concrete shape and type. In the future, a new API may be
// added to support cases requiring more advanced usage of the C.OrtTypeInfo
// struct.
func GetInputOutputInfo(path string) ([]InputOutputInfo, []InputOutputInfo,
	error) {
	s, e := createCSessionFromFile(path, nil)
	if e != nil {
		return nil, nil, fmt.Errorf("Error loading temporary session: %w", e)
	}
	defer C.ReleaseOrtSession(s)
	return getInputOutputInfoFromCSession(s)
}

// Identical in behavior to GetInputOutputInfo, but takes a slice of bytes
// containing the .onnx network rather than a file path.
func GetInputOutputInfoWithONNXData(data []byte) ([]InputOutputInfo,
	[]InputOutputInfo, error) {
	var e error
	s, e := createCSession(data, nil)
	if e != nil {
		return nil, nil, fmt.Errorf("Error creating temporary session: %w", e)
	}
	defer C.ReleaseOrtSession(s)
	return getInputOutputInfoFromCSession(s)
}

func getModelMetadataFromCSession(s *C.OrtSession) (*ModelMetadata, error) {
	var m *C.OrtModelMetadata
	status := C.SessionGetModelMetadata(s, &m)
	if status != nil {
		return nil, statusToError(status)
	}
	return &ModelMetadata{
		m: m,
	}, nil
}

// Takes a path to a .onnx file and returns the ModelMetadata associated with
// it. The returned metadata must be freed using its Destroy() function when
// it's no longer needed. InitializeEnvironment() must be called before using
// this function.
//
// Warning: This function loads the onnx content into a temporary onnxruntime
// session, so it may be computationally expensive.
func GetModelMetadata(path string) (*ModelMetadata, error) {
	s, e := createCSessionFromFile(path, nil)
	if e != nil {
		return nil, fmt.Errorf("Error loading %s: %w", path, e)
	}
	defer C.ReleaseOrtSession(s)
	return getModelMetadataFromCSession(s)
}

// Identical in behavior to GetModelMetadata, but takes a slice of bytes
// containing the .onnx network rather than a file path.
func GetModelMetadataWithONNXData(data []byte) (*ModelMetadata, error) {
	if !IsInitialized() {
		return nil, NotInitializedError
	}
	// Create the temporary ORT session from which we'll get the metadata.
	s, e := createCSession(data, nil)
	if e != nil {
		return nil, fmt.Errorf("Error creating temporary session: %w", e)
	}
	defer C.ReleaseOrtSession(s)
	return getModelMetadataFromCSession(s)
}
