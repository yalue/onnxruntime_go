package onnxruntime_go

// This file contains Session types that we maintain for compatibility
// purposes; the main onnxruntime_go.go file is dedicated to AdvancedSession
// now.

import (
	"fmt"
	"os"
)

// This type of session is for ONNX networks with the same input and output
// data types.
//
// NOTE: This type was written with a type parameter despite the fact that a
// type parameter is not necessary for any of its underlying implementation,
// which is a mistake in retrospect. It is preserved only for compatibility
// with older code, and new users should almost certainly be using an
// AdvancedSession instead.
//
// Using an AdvancedSession struct should be easier, and supports arbitrary
// combination of input and output tensor data types as well as more options.
type Session[T TensorData] struct {
	// We now delegate all of the implementation to an AdvancedSession here.
	s *AdvancedSession
}

// Similar to Session, but does not require the specification of the input
// and output shapes at session creation time, and allows for input and output
// tensors to have different types. This allows for fully dynamic input to the
// onnx model.
//
// NOTE: As with Session[T], new users should probably be using
// DynamicAdvancedSession in the future.
type DynamicSession[In TensorData, Out TensorData] struct {
	s *DynamicAdvancedSession
}

// The same as NewSession, but takes a slice of bytes containing the .onnx
// network rather than a file path.
func NewSessionWithONNXData[T TensorData](onnxData []byte, inputNames,
	outputNames []string, inputs, outputs []*Tensor[T]) (*Session[T], error) {
	// Unfortunately, a slice of pointers that satisfy an interface don't count
	// as a slice of interfaces (at least, as I write this), so we'll make the
	// conversion here.
	tmpInputs := make([]ArbitraryTensor, len(inputs))
	tmpOutputs := make([]ArbitraryTensor, len(outputs))
	for i, t := range inputs {
		tmpInputs[i] = t
	}
	for i, t := range outputs {
		tmpOutputs[i] = t
	}
	s, e := NewAdvancedSessionWithONNXData(onnxData, inputNames, outputNames,
		tmpInputs, tmpOutputs, nil)
	if e != nil {
		return nil, e
	}
	return &Session[T]{
		s: s,
	}, nil
}

// Similar to NewSessionWithOnnxData, but for dynamic sessions.
func NewDynamicSessionWithONNXData[in TensorData, out TensorData](onnxData []byte, inputNames, outputNames []string) (*DynamicSession[in, out], error) {
	s, e := NewDynamicAdvancedSessionWithONNXData(onnxData, inputNames,
		outputNames, nil)
	if e != nil {
		return nil, e
	}
	return &DynamicSession[in, out]{
		s: s,
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
func NewDynamicSession[in TensorData, out TensorData](onnxFilePath string,
	inputNames, outputNames []string) (*DynamicSession[in, out], error) {
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
	return s.s.Destroy()
}

func (s *DynamicSession[_, _]) Destroy() error {
	return s.s.Destroy()
}

func (s *Session[T]) Run() error {
	return s.s.Run()
}

// Unlike the non-dynamic equivalents, the DynamicSession's Run() function
// takes a list of input and output tensors rather than requiring the tensors
// to be specified at Session creation time. It is still the caller's
// responsibility to create and Destroy all tensors passed to this function.
func (s *DynamicSession[in, out]) Run(inputs []*Tensor[in],
	outputs []*Tensor[out]) error {
	if len(inputs) != len(s.s.s.inputs) {
		return fmt.Errorf("The session specified %d input names, but Run() "+
			"was called with %d input tensors", len(s.s.s.inputs), len(inputs))
	}
	if len(outputs) != len(s.s.s.outputs) {
		return fmt.Errorf("The session specified %d output names, but Run() "+
			"was called with %d output tensors", len(s.s.s.outputs),
			len(outputs))
	}

	// Rather than having to convert the Tensor pointers to ArbitraryTensor
	// types and calling GetInternals(), we'll just access the underlying
	// non-Dynamic AdvancedSession and set the inputs and outputs directly.
	for i, v := range inputs {
		s.s.s.inputs[i] = v.ortValue
	}
	for i, v := range outputs {
		s.s.s.outputs[i] = v.ortValue
	}
	return s.s.s.Run()
}
