package onnxruntime_go

// This file contains code and types that we maintain for compatibility
// purposes, but is not expected to be regularly maintained or udpated.

import (
	"fmt"
	"os"
)

// #include "onnxruntime_wrapper.h"
import "C"

// DEPRECATED: This type was written with a type parameter despite the fact
// that a type parameter is not necessary for any of its underlying
// implementation. It is preserved only for compatibility with older code, and
// new users should use AdvancedSession instead. Despite the name,
// AdvancedSession is equally simple to use and far more flexible.
type Session[T TensorData] struct {
	// We now delegate all of the implementation to an AdvancedSession here.
	s *AdvancedSession
}

// DEPRECATED: See the notes on Session[T]. Use DynamicAdvancedSession instead.
type DynamicSession[In TensorData, Out TensorData] struct {
	s *DynamicAdvancedSession
}

// DEPRECATED: See the notes on Session[T]. Use NewAdvancedSessionWithONNXData
// instead.
func NewSessionWithONNXData[T TensorData](onnxData []byte, inputNames,
	outputNames []string, inputs, outputs []*Tensor[T]) (*Session[T], error) {
	// Unfortunately, a slice of pointers that satisfy an interface don't count
	// as a slice of interfaces (at least, as I write this), so we'll make the
	// conversion here.
	tmpInputs := make([]Value, len(inputs))
	tmpOutputs := make([]Value, len(outputs))
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

// DEPRECATED: See the notes on Session[T]. Use
// NewDynamicAdvancedSessionWithONNXData instead.
func NewDynamicSessionWithONNXData[in TensorData, out TensorData](onnxData []byte,
	inputNames, outputNames []string) (*DynamicSession[in, out], error) {
	s, e := NewDynamicAdvancedSessionWithONNXData(onnxData, inputNames,
		outputNames, nil)
	if e != nil {
		return nil, e
	}
	return &DynamicSession[in, out]{
		s: s,
	}, nil
}

// DEPRECATED: See the notes on Session[T]. Use NewAdvancedSession instead.
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

// DEPRECATED: See the notes on Session[T]. Use NewDynamicAdvancedSession
// instead.
func NewDynamicSession[in TensorData, out TensorData](onnxFilePath string,
	inputNames, outputNames []string) (*DynamicSession[in, out], error) {
	fileContent, e := os.ReadFile(onnxFilePath)
	if e != nil {
		return nil, fmt.Errorf("Error reading %s: %w", onnxFilePath, e)
	}

	toReturn, e := NewDynamicSessionWithONNXData[in, out](fileContent,
		inputNames, outputNames)
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

func (s *DynamicSession[in, out]) Run(inputs []*Tensor[in],
	outputs []*Tensor[out]) error {
	if len(inputs) != len(s.s.s.inputNames) {
		return fmt.Errorf("The session specified %d input names, but Run() "+
			"was called with %d input tensors", len(s.s.s.inputNames),
			len(inputs))
	}
	if len(outputs) != len(s.s.s.outputNames) {
		return fmt.Errorf("The session specified %d output names, but Run() "+
			"was called with %d output tensors", len(s.s.s.outputNames),
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

	status := C.RunOrtSession(s.s.s.ortSession, &inputValues[0],
		&s.s.s.inputNames[0], C.int(len(inputs)), &outputValues[0],
		&s.s.s.outputNames[0], C.int(len(outputs)))
	if status != nil {
		return fmt.Errorf("Error running network: %w", statusToError(status))
	}
	return nil
}

// This type alias is included to avoid breaking older code, where the inputs
// and outputs to session.Run() were ArbitraryTensors rather than Values.
type ArbitraryTensor = Value

// As with the ArbitraryTensor type, this type alias only exists to facilitate
// renaming an old type without breaking existing code.
type TensorInternalData = ValueInternalData

var TrainingAPIRemovedError error = fmt.Errorf("Support for the training " +
	"API has been removed from onnxruntime_go following its deprecation in " +
	"onnxruntime versions 1.19.2 and later. The last revision of " +
	"onnxruntime_go supporting the training API is version v1.12.1")

// Support for TrainingSessions has been removed from onnxruntime_go following
// the deprecation of the training API in onnxruntime 1.20.0.
type TrainingSession struct{}

// Always returns TrainingAPIRemovedError.
func (s *TrainingSession) ExportModel(path string, outputNames []string) error {
	return TrainingAPIRemovedError
}

// Always returns TrainingAPIRemovedError.
func (s *TrainingSession) SaveCheckpoint(path string,
	saveOptimizerState bool) error {
	return TrainingAPIRemovedError
}

// Always returns TrainingAPIRemovedError.
func (s *TrainingSession) Destroy() error {
	return TrainingAPIRemovedError
}

// Always returns TrainingAPIRemovedError.
func (s *TrainingSession) TrainStep() error {
	return TrainingAPIRemovedError
}

// Always returns TrainingAPIRemovedError.
func (s *TrainingSession) OptimizerStep() error {
	return TrainingAPIRemovedError
}

// Always returns TrainingAPIRemovedError.
func (s *TrainingSession) LazyResetGrad() error {
	return TrainingAPIRemovedError
}

// Support for TrainingInputOutputNames has been removed from onnxruntime_go
// following the deprecation of the training API in onnxruntime 1.20.0.
type TrainingInputOutputNames struct {
	TrainingInputNames  []string
	EvalInputNames      []string
	TrainingOutputNames []string
	EvalOutputNames     []string
}

// Always returns (nil, TrainingAPIRemovedError).
func GetInputOutputNames(checkpointStatePath string, trainingModelPath string,
	evalModelPath string) (*TrainingInputOutputNames, error) {
	return nil, TrainingAPIRemovedError
}

// Always returns false.
func IsTrainingSupported() bool {
	return false
}

// Always returns (nil, TrainingAPIRemovedError).
func NewTrainingSessionWithOnnxData(checkpointData, trainingData, evalData,
	optimizerData []byte, inputs, outputs []Value,
	options *SessionOptions) (*TrainingSession, error) {
	return nil, TrainingAPIRemovedError
}

// Always returns (nil, TrainingAPIRemovedError).
func NewTrainingSession(checkpointStatePath, trainingModelPath, evalModelPath,
	optimizerModelPath string, inputs, outputs []Value,
	options *SessionOptions) (*TrainingSession, error) {
	return nil, TrainingAPIRemovedError
}
