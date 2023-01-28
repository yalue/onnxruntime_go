package onnxruntime

import (
	"encoding/json"
	"os"
	"runtime"
	"testing"
)

// This type is read from JSON and used to determine the inputs and expected
// outputs for an ONNX network.
type testInputsInfo struct {
	InputShape      []int     `json:input_shape`
	FlattenedInput  []float32 `json:flattened_input`
	OutputShape     []int     `json:output_shape`
	FlattenedOutput []float32 `json:flattened_output`
}

// This must be called prior to running each test.
func InitializeRuntime(t *testing.T) {
	if runtime.GOOS == "windows" {
		SetSharedLibraryPath("test_data/onnxruntime.dll")
	} else {
		if runtime.GOARCH == "arm64" {
			SetSharedLibraryPath("test_data/onnxruntime_arm64.so")
		} else {
			SetSharedLibraryPath("test_data/onnxruntime.so")
		}
	}
	e := InitializeEnvironment()
	if e != nil {
		t.Logf("Failed setting up onnxruntime environment: %s\n", e)
		t.FailNow()
	}
}

// Used to obtain the shape
func parseInputsJSON(path string, t *testing.T) *testInputsInfo {
	toReturn := testInputsInfo{}
	f, e := os.Open(path)
	if e != nil {
		t.Logf("Failed opening %s: %s\n", path, e)
		t.FailNow()
	}
	defer f.Close()
	d := json.NewDecoder(f)
	e = d.Decode(&toReturn)
	if e != nil {
		t.Logf("Failed decoding %s: %s\n", path, e)
		t.FailNow()
	}
	return &toReturn
}

func TestExampleNetwork(t *testing.T) {
	InitializeRuntime(t)
	_ = parseInputsJSON("test_data/example_network_results.json", t)

	// TODO: More tests here to run the network, once that's supported.

	e := CleanupEnvironment()
	if e != nil {
		t.Logf("Failed cleaning up the environment: %s\n", e)
		t.FailNow()
	}
}
