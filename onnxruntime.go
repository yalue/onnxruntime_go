// This library wraps the C "onnxruntime" library maintained at
// https://github.com/microsoft/onnxruntime.  It seeks to provide as simple an
// interface as possible to load and run ONNX-format neural networks from
// Go code.
package onnxruntime

import (
	"fmt"
	"os"
	"unsafe"
)

// #cgo CFLAGS: -I${SRCDIR}/onnxruntime/include
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

// Call this function to initialize the internal onnxruntime environment. If
// this doesn't return an error, the caller will be responsible for calling
// CleanupEnvironment to free the onnxruntime state when no longer needed.
func InitializeEnvironment() error {
	if ortEnv != nil {
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
		CleanupEnvironment()
		return fmt.Errorf("Error creating ORT memory info: %w",
			statusToError(status))
	}

	return nil
}

// Call this function to cleanup the internal onnxruntime environment when it
// is no longer needed.
func CleanupEnvironment() error {
	var e error
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

// The Shape type holds the shape of the tensors used by the network input and
// outputs.
type Shape []int64

// Returns a Shape, with the given dimensions.
func NewShape(dimensions ...int64) Shape {
	return Shape(dimensions)
}

// Returns the total number of elements in a tensor with the given shape.
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

// Makes and returns a deep copy of the Shape.
func (s Shape) Clone() Shape {
	toReturn := make([]int64, len(s))
	copy(toReturn, []int64(s))
	return Shape(toReturn)
}

func (s Shape) String() string {
	return fmt.Sprintf("%v", []int64(s))
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
func (t *Tensor[_]) Destroy() {
	C.ReleaseOrtValue(t.ortValue)
	t.ortValue = nil
	t.data = nil
	t.shape = nil
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
// returned by this function must be destroyed when no longer needed.
func (t *Tensor[T]) Clone() (*Tensor[T], error) {
	// TODO: Implement Tensor.Clone()
	return nil, fmt.Errorf("Not yet implemented")
}

// Creates a new empty tensor with the given shape. The shape provided to this
// function is copied, and is no longer needed after this function returns.
func NewEmptyTensor[T TensorData](s Shape) (*Tensor[T], error) {
	elementCount := s.FlattenedSize()
	if elementCount == 0 {
		return nil, fmt.Errorf("Got invalid shape containing 0 elements")
	}
	data := make([]T, elementCount)
	return NewTensor(s, data)
}

// Creates a new tensor backed by an existing data slice. The shape provided to
// this function is copied, and is no longer needed after this function
// returns. If the data slice is longer than s.FlattenedSize(), then only the
// first portion of the data will be used.
func NewTensor[T TensorData](s Shape, data []T) (*Tensor[T], error) {
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
	// TODO (next): Set a finalizer on new Tensors.
	// - Idea: use a "destroyable" interface
	return &toReturn, nil
}

// A simple wrapper around the OrtSession C struct. Requires the user to
// maintain all input and output tensors, and to use the same data type for
// input and output tensors.
type SimpleSession[T TensorData] struct {
	ortSession *C.OrtSession
}

// Loads the ONNX network at the given path, and initializes a SimpleSession
// instance. If this returns successfully, the caller must call Destroy() on
// the returned session when it is no longer needed.
func NewSimpleSession[T TensorData](onnxFilePath string) (*SimpleSession[T],
	error) {
	// We load content this way in order to avoid a mess of wide-character
	// paths on Windows if we use CreateSession rather than
	// CreateSessionFromArray.
	fileContent, e := os.ReadFile(onnxFilePath)
	if e != nil {
		return nil, fmt.Errorf("Error reading %s: %w", onnxFilePath, e)
	}
	var ortSession *C.OrtSession
	status := C.CreateSimpleSession(unsafe.Pointer(&(fileContent[0])),
		C.size_t(len(fileContent)), ortEnv, &ortSession)
	if status != nil {
		return nil, fmt.Errorf("Error creating session from %s: %w",
			onnxFilePath, statusToError(status))
	}
	// ONNXRuntime copies the file content unless a specific flag is provided
	// when creating the session (and we don't provide it!)
	fileContent = nil
	return &SimpleSession[T]{
		ortSession: ortSession,
	}, nil
}

func (s *SimpleSession[_]) Destroy() error {
	if s.ortSession != nil {
		C.ReleaseOrtSession(s.ortSession)
		s.ortSession = nil
	}
	return nil
}

// This function assumes the SimpleSession takes a single input tensor and
// produces a single output, both of which have the same type.
func (s *SimpleSession[T]) SimpleRun(input *Tensor[T],
	output *Tensor[T]) error {
	status := C.RunSimpleSession(s.ortSession, input.ortValue,
		output.ortValue)
	if status != nil {
		return fmt.Errorf("Error running network: %w", statusToError(status))
	}
	return nil
}

// TODO (next): Test SimpleRun
