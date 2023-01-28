// This application loads a test ONNX network and executes it on some fixed
// data. It serves as an example of how to use the onnxruntime wrapper library.
package main

import (
	"fmt"
	"github.com/yalue/onnxruntime"
	"os"
	"runtime"
)

func run() int {
	if runtime.GOOS == "windows" {
		onnxruntime.SetSharedLibraryPath("../test_data/onnxruntime.dll")
	} else {
		onnxruntime.SetSharedLibraryPath("../test_data/onnxruntime.so")
	}
	e := onnxruntime.InitializeEnvironment()
	if e != nil {
		fmt.Printf("Error initializing the onnxruntime environment: %s\n", e)
		return 1
	}
	fmt.Printf("The onnxruntime environment initialized OK.\n")

	// Ordinarily, it is probably fine to call this using defer, but we do it
	// here just so we can print a status message after the cleanup completes.
	e = onnxruntime.CleanupEnvironment()
	if e != nil {
		fmt.Printf("Error cleaning up the environment: %s\n", e)
		return 1
	}
	fmt.Printf("The onnxruntime environment was cleaned up OK.\n")
	return 0
}

func main() {
	os.Exit(run())
}
