package onnxruntime_go

import (
	"errors"
	"math"
	"math/rand"
	"os"
	"path"
	"testing"
)

func TestTrainingNotSupported(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)

	if IsTrainingSupported() {
		t.Skipf("onnxruntime library supports training")
	}

	options, e := NewSessionOptions()
	if e != nil {
		t.Logf("Failed creating options: %s\n", e)
		t.FailNow()
	}

	trainingSession, e := NewTrainingSession("test_data/onnxruntime_training_test/training_artifacts/checkpoint",
		"test_data/onnxruntime_training_test/training_artifacts/training_model.onnx",
		"test_data/onnxruntime_training_test/training_artifacts/eval_model.onnx",
		"test_data/onnxruntime_training_test/training_artifacts/optimizer_model.onnx",
		nil, nil,
		options)

	if !errors.Is(e, trainingNotSupportedError) {
		t.Logf("Creating training session when onnxruntime lib does not support it should return training not supported error.")
		if e != nil {
			t.Logf("Received instead error: %s", e.Error())
		} else {
			t.Logf("Received no error instead")
		}
		t.FailNow()
	}
	if trainingSession != nil {
		if err := trainingSession.Destroy(); err != nil {
			t.Fatalf("cleanup of training session failed with error: %v", e)
		}
	}
}

func TestGetInputOutputNames(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)

	if !IsTrainingSupported() {
		t.Skipf("Training is not supported on this platform/onnxruntime build.")
	}

	artifactsPath := path.Join("test_data", "training_test")

	names, err := GetInputOutputNames(
		path.Join(artifactsPath, "checkpoint"),
		path.Join(artifactsPath, "training_model.onnx"),
		path.Join(artifactsPath, "eval_model.onnx"),
	)

	if err != nil {
		t.Fatalf("Failed getting input and output names with error: %v\n", err)
	}

	expectedTrainInputNames := []string{"input", "target"}
	expectedEvalInputNames := expectedTrainInputNames
	expectedTrainOutputNames := []string{"onnx::reducemean_output::5"}
	expectedEvalOutputNames := expectedTrainOutputNames

	for i, v := range names.TrainingInputNames {
		if v != expectedTrainInputNames[i] {
			t.Fatalf("training input names don't match")
		}
	}
	for i, v := range names.TrainingOutputNames {
		if v != expectedTrainOutputNames[i] {
			t.Fatalf("training output names don't match")
		}
	}
	for i, v := range names.EvalInputNames {
		if v != expectedEvalInputNames[i] {
			t.Fatalf("eval input names don't match")
		}
	}
	for i, v := range names.EvalOutputNames {
		if v != expectedEvalOutputNames[i] {
			t.Fatalf("eval output names don't match")
		}
	}

	// without eval model
	names, err = GetInputOutputNames(
		path.Join(artifactsPath, "checkpoint"),
		path.Join(artifactsPath, "training_model.onnx"),
		"",
	)

	if err != nil {
		t.Fatalf("Failed getting input and output names with error: %v\n", err)
	}

	for i, v := range names.TrainingInputNames {
		if v != expectedTrainInputNames[i] {
			t.Fatalf("training input names don't match")
		}
	}
	for i, v := range names.TrainingOutputNames {
		if v != expectedTrainOutputNames[i] {
			t.Fatalf("training output names don't match")
		}
	}
}

func generateBatchData(nBatches int, batchSize int) map[int]map[string][]float32 {
	batchData := map[int]map[string][]float32{}

	source := rand.NewSource(1234)
	g := rand.New(source)

	for i := 0; i < nBatches; i++ {
		inputCounter := 0
		outputCounter := 0
		inputSlice := make([]float32, batchSize*4)
		outputSlice := make([]float32, batchSize*2)
		batchData[i] = map[string][]float32{}

		// generate random data for batch
		for n := 0; n < batchSize; n++ {
			var sum float32
			min := float32(1)
			max := float32(-1)
			for i := 0; i < 4; i++ {
				r := g.Float32()
				inputSlice[inputCounter] = r
				inputCounter++
				if r > max {
					max = r
				}
				if r < min {
					min = r
				}
				sum = sum + r
			}
			outputSlice[outputCounter] = sum
			outputSlice[outputCounter+1] = max - min
			outputCounter = outputCounter + 2
		}
		batchData[i]["input"] = inputSlice
		batchData[i]["output"] = outputSlice
	}
	return batchData
}

// TestTraining tests a basic training flow using the bindings to the C api for on-device onnxruntime training
func TestTraining(t *testing.T) {
	InitializeRuntime(t)
	defer CleanupRuntime(t)

	if !IsTrainingSupported() {
		t.Skipf("Training is not supported on this platform/onnxruntime build.")
	}

	trainingArtifactsFolder := path.Join("test_data", "training_test")

	// generate training data
	batchSize := 10
	nBatches := 10

	// holds inputs/outputs and loss for each training batch
	batchInputShape := NewShape(int64(batchSize), 1, 4)
	batchTargetShape := NewShape(int64(batchSize), 1, 2)
	batchInputTensor, err := NewEmptyTensor[float32](batchInputShape)
	if err != nil {
		t.Fatalf("training test failed with error: %v", err)
	}
	batchTargetTensor, err := NewEmptyTensor[float32](batchTargetShape)
	if err != nil {
		t.Fatalf("training test failed with error: %v", err)
	}
	lossScalar, err := NewEmptyScalar[float32]()
	if err != nil {
		t.Fatalf("training test failed with error: %v", err)
	}

	trainingSession, errorSessionCreation := NewTrainingSession(
		path.Join(trainingArtifactsFolder, "checkpoint"),
		path.Join(trainingArtifactsFolder, "training_model.onnx"),
		path.Join(trainingArtifactsFolder, "eval_model.onnx"),
		path.Join(trainingArtifactsFolder, "optimizer_model.onnx"),
		[]Value{batchInputTensor, batchTargetTensor}, []Value{lossScalar},
		nil)

	if errorSessionCreation != nil {
		t.Fatalf("session creation failed with error: %v", errorSessionCreation)
	}

	// cleanup after test run
	defer func(session *TrainingSession, tensors []Value) {
		var errs []error
		errs = append(errs, session.Destroy())
		for _, t := range tensors {
			errs = append(errs, t.Destroy())
		}
		if e := errors.Join(errs...); e != nil {
			t.Fatalf("cleanup of test failed with error: %v", e)
		}
	}(trainingSession, []Value{batchInputTensor, batchTargetTensor, lossScalar})

	losses := []float32{}
	epochs := 100
	batchData := generateBatchData(nBatches, batchSize)

	for epoch := 0; epoch < epochs; epoch++ {
		var epochLoss float32 // total epoch loss

		for i := 0; i < nBatches; i++ {
			inputSlice := batchInputTensor.GetData()
			outputSlice := batchTargetTensor.GetData()

			copy(inputSlice, batchData[i]["input"])
			copy(outputSlice, batchData[i]["output"])

			// train on batch
			err = trainingSession.TrainStep()
			if err != nil {
				t.Fatalf("train step failed with error: %v", err)
			}

			epochLoss = epochLoss + lossScalar.GetData()

			err = trainingSession.OptimizerStep()
			if err != nil {
				t.Fatalf("optimizer step failed with error: %v", err)
			}

			// ort training api - reset the gradients to zero so that new gradients can be computed for next batch
			err = trainingSession.LazyResetGrad()
			if err != nil {
				t.Fatalf("lazy reset grad step failed with error: %v", err)
			}
		}
		if epoch%10 == 0 {
			t.Logf("Epoch {%d} Loss {%f}\n", epoch+1, epochLoss/float32(batchSize*nBatches))
			losses = append(losses, epochLoss/float32(batchSize*nBatches))
		}
	}

	expectedLosses := []float32{
		0.125085,
		0.097187,
		0.062333,
		0.024307,
		0.019963,
		0.018476,
		0.017160,
		0.015982,
		0.014845,
		0.013867,
	}

	for i, l := range losses {
		diff := math.Abs(float64(l - expectedLosses[i]))
		deviation := diff / float64(expectedLosses[i])
		if deviation > 0.6 {
			t.Fatalf("loss deviation too large: expected %f, actual %f, deviation %f", float64(expectedLosses[i]), float64(l), float64(deviation))
		}
	}

	// test the saving of the checkpoint state
	finalCheckpointPath := path.Join("test_data", "training_test", "finalCheckpoint")
	errSaveCheckpoint := trainingSession.SaveCheckpoint(finalCheckpointPath, false)
	if errSaveCheckpoint != nil {
		t.Fatalf("Saving of checkpoint failed with error: %v", errSaveCheckpoint)
	}

	// test the saving of the model
	finalModelPath := path.Join("test_data", "training_test", "final_inference.onnx")
	errExport := trainingSession.ExportModel(finalModelPath, []string{"output"})
	if errExport != nil {
		t.Fatalf("Exporting model failed with error: %v", errExport)
	}

	defer func() {
		e := os.Remove(finalCheckpointPath)
		if e != nil {
			t.Errorf("Error removing final checkpoint file %s: %s", finalCheckpointPath, e)
		}
		e = os.Remove(finalModelPath)
		if e != nil {
			t.Errorf("Error removing final model file %s: %s", finalModelPath, e)
		}
	}()

	// load the model back in and test in-sample predictions for the first batch
	// (we care about correctness more than generalization here)
	session, err := NewAdvancedSession(path.Join("test_data", "training_test", "final_inference.onnx"),
		[]string{"input"}, []string{"output"},
		[]Value{batchInputTensor}, []Value{batchTargetTensor}, nil)

	if err != nil {
		t.Fatalf("creation of inference session failed with error: %v", err)
	}

	defer func(s *AdvancedSession) {
		err := s.Destroy()
		if err != nil {
			t.Fatalf("cleanup of inference session failed with error: %v", err)
		}
	}(session)

	// Calling Run() will run the network, reading the current contents of the
	// input tensors and modifying the contents of the output tensors.
	copy(batchInputTensor.GetData(), batchData[0]["input"])
	err = session.Run()
	if err != nil {
		t.Fatalf("run of inference session failed with error: %v", err)
	}

	expectedOutput := []float32{
		2.4524384,
		0.65120333,
		2.5457804,
		0.6102175,
		1.6276635,
		0.573755,
		1.7900972,
		0.59951085,
		3.1650176,
		0.66626525,
		1.9361509,
		0.571084,
		2.0798547,
		0.6060241,
		0.9611889,
		0.52100605,
		1.4070896,
		0.5412475,
		2.1449144,
		0.5985652,
	}

	for i, l := range batchTargetTensor.GetData() {
		diff := math.Abs(float64(l - expectedOutput[i]))
		deviation := diff / float64(expectedOutput[i])
		if deviation > 0.6 {
			t.Fatalf("deviation too large")
		}
	}
}
