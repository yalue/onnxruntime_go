package onnxruntime_go

import "C"
import (
	"fmt"
	"os"
	"unsafe"
)

type OrtSession struct {
	Session *C.OrtSession
}

func (o *OrtSession) CreateCSessionSave(modelPath string, options *SessionOptions) error {
	if !IsInitialized() {
		return NotInitializedError
	}

	var ortSessionOptions *C.OrtSessionOptions
	if options != nil {
		ortSessionOptions = options.o
	}

	onnxData, e := os.ReadFile(modelPath)
	if e != nil {
		return fmt.Errorf("Error reading %s: %w", modelPath, e)
	}

	status := C.CreateSession(unsafe.Pointer(&(onnxData[0])),
		C.size_t(len(onnxData)), ortEnv, &o.Session, ortSessionOptions)

	if status != nil {
		return statusToError(status)
	}
	return nil
}

// The same as NewAdvancedSession, but takes a slice of bytes containing the
// .onnx network rather than a file path.
func (o *OrtSession) NewAdvancedSessionWithONNXData(inputNames,
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
		ortSession:  o.Session,
		inputNames:  cInputNames,
		outputNames: cOutputNames,
		inputs:      inputOrtTensors,
		outputs:     outputOrtTensors,
	}, nil
}
