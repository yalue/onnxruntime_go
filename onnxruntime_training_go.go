package onnxruntime_go

// #cgo CFLAGS: -O2 -g
//
// #include "onnxruntime_wrapper.h"
import "C"
import (
	"fmt"
	"os"
	"path/filepath"
	"unsafe"
)

var trainingNotSupportedError error = fmt.Errorf("Training not supported by onnx library")

// Scalar is like a tensor but the underlying go slice is of length 1 and it has no dimension.
// It can be used to store e.g. the loss from a training cycle.
type Scalar[T TensorData] struct {
	data     []T
	dataSize uintptr
	ortValue *C.OrtValue
}

func (s *Scalar[T]) GetShape() Shape {
	return nil
}

func (s *Scalar[T]) ZeroContents() {
	C.memset(unsafe.Pointer(&s.data[0]), 0, C.size_t(s.dataSize))
}

func (s *Scalar[T]) Destroy() error {
	C.ReleaseOrtValue(s.ortValue)
	s.ortValue = nil
	s.data = nil
	s.dataSize = 0
	return nil
}

// GetData returns the undelying data for the scalar.
// If you want to explicitly set the scalar's data, use Set.
func (t *Scalar[T]) GetData() T {
	return t.data[0]
}

// Set allows to explicitly set the underlying value for the scalar.
func (t *Scalar[T]) Set(value T) {
	t.data = []T{value}
}

func (t *Scalar[T]) DataType() C.ONNXTensorElementDataType {
	return GetTensorElementDataType[T]()
}

func (t *Scalar[_]) GetInternals() *TensorInternalData {
	return &TensorInternalData{
		ortValue: t.ortValue,
	}
}

// NewEmptyScalar creates a new scalar of type T.
func NewEmptyScalar[T TensorData]() (*Scalar[T], error) {
	var data T
	return NewScalar(data)
}

// NewScalar creates a new scalar of type T backed by a value of type T.
// Note that, differently from tensors, this is not a []T but just a value T.
func NewScalar[T TensorData](data T) (*Scalar[T], error) {
	if !IsInitialized() {
		return nil, NotInitializedError
	}

	dataSlice := []T{data}
	var ortValue *C.OrtValue
	dataType := GetTensorElementDataType[T]()
	dataSize := unsafe.Sizeof(dataSlice[0]) * uintptr(1)

	status := C.CreateOrtTensorWithShape(unsafe.Pointer(&dataSlice[0]),
		C.size_t(dataSize), nil, C.int64_t(0), ortMemoryInfo, dataType, &ortValue)
	if status != nil {
		return nil, statusToError(status)
	}
	toReturn := Scalar[T]{
		data:     dataSlice,
		dataSize: dataSize,
		ortValue: ortValue,
	}
	return &toReturn, nil
}

// TraininSession is the type that wraps the C training session object.
type TrainingSession struct {
	ortTrainingSession *C.OrtTrainingSession
	ortCheckpointState *C.OrtCheckpointState
	inputs             []*C.OrtValue
	outputs            []*C.OrtValue
	trainingModelPath  *C.char
	optimizerModelPath *C.char
	evalModelPath      *C.char
}

// ExportModel is used to export the final trained model to disk. It requires the path for
// the exported model as well as the names of the graph nodes to export.
// Note that currently the final model can only be exported if the session has been
// initialized with NewTrainingSession and the path to the eval model has been provided.
func (s *TrainingSession) ExportModel(path string, outputNames []string) error {
	if s.evalModelPath == nil {
		return fmt.Errorf(`final model can only be exported if the eval model path is 
		provided at session creation time (see NewTrainingSession)`)
	}
	if path == "" {
		return fmt.Errorf("Path cannot be empty")
	}
	dir, _ := filepath.Split(path)
	if _, err := os.Stat(dir); dir != "" && os.IsNotExist(err) {
		return fmt.Errorf("Directory %s does not exist", dir)
	}

	cOutputNames := make([]*C.char, len(outputNames))
	for i, name := range outputNames {
		cOutputNames[i] = C.CString(name)
	}
	cPath := C.CString(path)
	outputLength := C.size_t(len(outputNames))
	defer func() {
		for i := range cOutputNames {
			C.free(unsafe.Pointer(cOutputNames[i]))
		}
		C.free(unsafe.Pointer(cPath))
	}()
	status := C.ExportModel(s.ortTrainingSession, cPath, outputLength, &cOutputNames[0])
	if status != nil {
		return statusToError(status)
	}
	return nil
}

// SaveCheckpoint can be used to save the current checkpoint state at the specified path.
// This is useful to snapshot the training parameters to continue training later or on
// a different machine.
func (s *TrainingSession) SaveCheckpoint(path string, saveOptimizerState bool) error {
	if path == "" {
		return fmt.Errorf("Path cannot be empty")
	}
	dir, _ := filepath.Split(path)
	if _, err := os.Stat(dir); dir != "" && os.IsNotExist(err) {
		return fmt.Errorf("Directory %s does not exist", dir)
	}

	cPath := C.CString(path)
	var saveOptimizer int
	if saveOptimizerState {
		saveOptimizer = 1
	}

	defer func() {
		C.free(unsafe.Pointer(cPath))
	}()

	status := C.SaveCheckpoint(s.ortCheckpointState, cPath, C.size_t(saveOptimizer))
	if status != nil {
		return statusToError(status)
	}
	return nil
}

// Destroy frees all the C memory associated to a training session.
func (s *TrainingSession) Destroy() error {
	if s.ortTrainingSession != nil {
		C.ReleaseOrtTrainingSession(s.ortTrainingSession)
		s.ortTrainingSession = nil
	}
	// note: checkpoint MUST be released after session
	if s.ortCheckpointState != nil {
		C.ReleaseCheckpointState(s.ortCheckpointState)
		s.ortCheckpointState = nil
	}
	C.free(unsafe.Pointer(s.trainingModelPath))
	s.trainingModelPath = nil
	C.free(unsafe.Pointer(s.evalModelPath))
	s.evalModelPath = nil
	C.free(unsafe.Pointer(s.optimizerModelPath))
	s.optimizerModelPath = nil
	s.inputs = nil
	s.outputs = nil
	return nil
}

// TrainStep performs the training step.
func (s *TrainingSession) TrainStep() error {
	inputLength := C.size_t(len(s.inputs))
	outputLength := C.size_t(len(s.outputs))
	status := C.TrainStep(s.ortTrainingSession, inputLength, &s.inputs[0], outputLength, &s.outputs[0])
	if status != nil {
		return fmt.Errorf("Error performing training step: %w", statusToError(status))
	}
	return nil
}

// TrainStep performs the optimizer step.
func (s *TrainingSession) OptimizerStep() error {
	status := C.OptimizerStep(s.ortTrainingSession)
	if status != nil {
		return fmt.Errorf("Error performing optimizer step: %w", statusToError(status))
	}
	return nil
}

// TrainStep performs the LazyResetGrad step.
func (s *TrainingSession) LazyResetGrad() error {
	status := C.LazyResetGrad(s.ortTrainingSession)
	if status != nil {
		return fmt.Errorf("Error performing lazyResetGrad step: %w", statusToError(status))
	}
	return nil
}

func getInputName(s *C.OrtTrainingSession, i int, model string) (string, error) {
	var cName *C.char
	var status *C.OrtStatus
	switch model {
	case "train":
		status = C.TrainingSessionGetTrainingInputName(s, C.size_t(i), &cName)
	case "eval":
		status = C.TrainingSessionGetEvalInputName(s, C.size_t(i), &cName)
	default:
		return "", fmt.Errorf("%s model not recognized", model)
	}
	if status != nil {
		return "", fmt.Errorf("Error getting name: %w", statusToError(status))
	}

	name, e := convertORTString(cName)
	if e != nil {
		return "", fmt.Errorf("Error converting C name to Go string: %w", e)
	}
	return name, nil
}

func getOutputName(s *C.OrtTrainingSession, i int, model string) (string, error) {
	var cName *C.char
	var status *C.OrtStatus
	switch model {
	case "train":
		status = C.TrainingSessionGetTrainingOutputName(s, C.size_t(i), &cName)
	case "eval":
		status = C.TrainingSessionGetEvalOutputName(s, C.size_t(i), &cName)
	default:
		return "", fmt.Errorf("%s model not recognized", model)
	}
	if status != nil {
		return "", fmt.Errorf("Error getting name: %w", statusToError(status))
	}
	name, e := convertORTString(cName)
	if e != nil {
		return "", fmt.Errorf("Error converting C name to Go string: %w", e)
	}
	return name, nil
}

type TrainingInputOutputNames struct {
	TrainingInputNames  []string
	EvalInputNames      []string
	TrainingOutputNames []string
	EvalOutputNames     []string
}

// GetInputOutputNames returns the names of the training inputs and outputs
// for each model.
func (s *TrainingSession) GetInputOutputNames() (*TrainingInputOutputNames, error) {
	var inputCountTraining, inputCountEval C.size_t
	status := C.TrainingSessionGetInputCount(s.ortTrainingSession, &inputCountTraining, &inputCountEval)
	if status != nil {
		return nil, statusToError(status)
	}

	var outputCountTraining, outputCountEval C.size_t
	status = C.TrainingSessionGetOutputCount(s.ortTrainingSession, &outputCountTraining, &outputCountEval)
	if status != nil {
		return nil, statusToError(status)
	}

	trainInputNames := make([]string, inputCountTraining)
	trainOutputNames := make([]string, outputCountTraining)
	evalInputNames := make([]string, inputCountEval)
	evalOutputNames := make([]string, outputCountEval)

	for i := 0; i < int(inputCountTraining); i++ {
		name, err := getInputName(s.ortTrainingSession, i, "train")
		if err != nil {
			return nil, fmt.Errorf("Error retrieving train input name: %w", err)
		}
		trainInputNames[i] = name
	}

	for i := 0; i < int(inputCountEval); i++ {
		name, err := getInputName(s.ortTrainingSession, i, "eval")
		if err != nil {
			return nil, fmt.Errorf("Error retrieving eval input name: %w", err)
		}
		evalInputNames[i] = name
	}

	for i := 0; i < int(outputCountTraining); i++ {
		name, err := getOutputName(s.ortTrainingSession, i, "train")
		if err != nil {
			return nil, fmt.Errorf("Error retrieving train output name: %w", err)
		}
		trainOutputNames[i] = name
	}

	for i := 0; i < int(outputCountTraining); i++ {
		name, err := getOutputName(s.ortTrainingSession, i, "eval")
		if err != nil {
			return nil, fmt.Errorf("Error retrieving eval output name: %w", err)
		}
		evalOutputNames[i] = name
	}

	return &TrainingInputOutputNames{
		TrainingInputNames:  trainInputNames,
		EvalInputNames:      evalInputNames,
		TrainingOutputNames: trainOutputNames,
		EvalOutputNames:     evalOutputNames,
	}, nil
}

// IsTrainingSupported returns true if the training api is supported
// by the onnxruntime library.
func IsTrainingSupported() bool {
	isApiSupported := C.IsTrainingApiSupported()
	if isApiSupported == 1 {
		return true
	} else {
		return false
	}
}

func checkTraining() error {
	if !IsInitialized() {
		return NotInitializedError
	}
	if !IsTrainingSupported() {
		return trainingNotSupportedError
	}
	return nil
}

func createCCheckpoint(onnxData []byte) (*C.OrtCheckpointState, error) {
	if e := checkTraining(); e != nil {
		return nil, e
	}
	if len(onnxData) == 0 {
		return nil, fmt.Errorf("Missing checkpoint data")
	}
	var ortCheckpointState *C.OrtCheckpointState
	status := C.CreateCheckpoint(unsafe.Pointer(&(onnxData[0])), C.size_t(len(onnxData)), &ortCheckpointState)
	if status != nil {
		return nil, statusToError(status)
	}
	return ortCheckpointState, nil
}

// createCTrainingSessionWithOnnxData creates a C session from byte data using buffers
func createCTrainingSessionWithOnnxData(checkpointState *C.OrtCheckpointState,
	trainingData, evalData, optimizerData []byte,
	options *SessionOptions) (*C.OrtTrainingSession, error) {
	if e := checkTraining(); e != nil {
		return nil, e
	}
	var ortTrainingSession *C.OrtTrainingSession
	var ortSessionOptions *C.OrtSessionOptions
	if options != nil {
		ortSessionOptions = options.o
	}
	status := C.CreateTrainingSessionFromBuffer(
		checkpointState,
		unsafe.Pointer(&(trainingData[0])), C.size_t(len(trainingData)),
		unsafe.Pointer(&(evalData[0])), C.size_t(len(evalData)),
		unsafe.Pointer(&(optimizerData[0])), C.size_t(len(optimizerData)),
		ortEnv, &ortTrainingSession, ortSessionOptions)
	if status != nil {
		return nil, statusToError(status)
	}
	return ortTrainingSession, nil
}

// createCTrainingSessionWithPaths creates a C session from paths
func createCtrainingSessionWithPaths(checkpointState *C.OrtCheckpointState,
	trainingPath, evalPath, optimizerPath *C.char,
	options *SessionOptions) (*C.OrtTrainingSession, error) {
	if e := checkTraining(); e != nil {
		return nil, e
	}

	var ortTrainingSession *C.OrtTrainingSession
	var ortSessionOptions *C.OrtSessionOptions
	if options != nil {
		ortSessionOptions = options.o
	}

	status := C.CreateTrainingSessionFromPaths(checkpointState,
		trainingPath, evalPath, optimizerPath, ortEnv, &ortTrainingSession, ortSessionOptions)

	if status != nil {
		return nil, statusToError(status)
	}
	return ortTrainingSession, nil
}

// NewTrainingSessionWithOnnxData is like NewTrainingSession, but it accepts
// bytes rather than paths to the training assets. Note that there does not
// seem to currently be a way to export the trained model from a session instantiated
// from bytes. If you wish to export the trained model, you should use NewTrainingSession
// instead.
func NewTrainingSessionWithOnnxData(checkpointData []byte,
	trainingData []byte,
	evalData []byte,
	optimizerData []byte,
	inputs []ArbitraryTensor, outputs []ArbitraryTensor,
	options *SessionOptions) (*TrainingSession, error) {

	err := checkTraining()
	if err != nil {
		return nil, err
	}

	// create checkpoint C object
	ortCheckpointState, e := createCCheckpoint(checkpointData)
	if e != nil {
		return nil, fmt.Errorf("Error creating C checkpointState: %w", e)
	}

	// create session C object
	ortTrainingSession, e := createCTrainingSessionWithOnnxData(ortCheckpointState,
		trainingData, evalData, optimizerData, options)
	if e != nil {
		return nil, fmt.Errorf("Error creating C training session: %w", e)
	}

	inputOrtTensors := make([]*C.OrtValue, len(inputs))
	outputOrtTensors := make([]*C.OrtValue, len(outputs))
	for i, v := range inputs {
		inputOrtTensors[i] = v.GetInternals().ortValue
	}
	for i, v := range outputs {
		outputOrtTensors[i] = v.GetInternals().ortValue
	}

	return &TrainingSession{
		ortCheckpointState: ortCheckpointState,
		ortTrainingSession: ortTrainingSession,
		inputs:             inputOrtTensors,
		outputs:            outputOrtTensors,
	}, nil
}

// NewTrainingSession creates a new training session from paths stored on disk.
// evalModelPath is optional and can be the empty string. In case it is not
// provided, only the checkpoint state can be exported once training is complete
// (and not the final inference model).
func NewTrainingSession(checkpointStatePath string,
	trainingModelPath string,
	evalModelPath string,
	optimizerModelPath string,
	inputs,
	outputs []ArbitraryTensor,
	options *SessionOptions) (*TrainingSession, error) {

	err := checkTraining()
	if err != nil {
		return nil, err
	}

	checkPointContent, e := os.ReadFile(checkpointStatePath)
	if e != nil {
		return nil, fmt.Errorf("Reading checkpoint data failed: %s", e.Error())
	}

	// create checkpoint C object
	ortCheckpointState, e := createCCheckpoint(checkPointContent)
	if e != nil {
		return nil, fmt.Errorf("Error creating C checkpointState: %w", e)
	}

	// create session C object
	if _, err := os.Stat(trainingModelPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("training model does not exist at path %s", trainingModelPath)
	}
	cTrainingPath := C.CString(trainingModelPath)

	// eval is optional
	var cEvalPath *C.char
	if evalModelPath != "" {
		if _, err := os.Stat(evalModelPath); os.IsNotExist(err) {
			return nil, fmt.Errorf("eval model does not exist at path %s", evalModelPath)
		}
		cEvalPath = C.CString(evalModelPath)
	}

	if _, err := os.Stat(optimizerModelPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("optimizer s does not exist at path %s", optimizerModelPath)
	}
	COptimizerPath := C.CString(optimizerModelPath)

	ortTrainingSession, e := createCtrainingSessionWithPaths(ortCheckpointState,
		cTrainingPath, cEvalPath, COptimizerPath, options)
	if e != nil {
		return nil, fmt.Errorf("Error creating C training session: %w", e)
	}

	inputOrtTensors := make([]*C.OrtValue, len(inputs))
	outputOrtTensors := make([]*C.OrtValue, len(outputs))
	for i, v := range inputs {
		inputOrtTensors[i] = v.GetInternals().ortValue
	}
	for i, v := range outputs {
		outputOrtTensors[i] = v.GetInternals().ortValue
	}

	return &TrainingSession{
		ortCheckpointState: ortCheckpointState,
		ortTrainingSession: ortTrainingSession,
		inputs:             inputOrtTensors,
		outputs:            outputOrtTensors,
		evalModelPath:      cEvalPath,
		trainingModelPath:  cTrainingPath,
		optimizerModelPath: COptimizerPath,
	}, nil
}
