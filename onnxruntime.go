// This library wraps the C "onnxruntime" library maintained at
// https://github.com/microsoft/onnxruntime.  It seeks to provide as simple an
// interface as possible to load and run ONNX-format neural networks from
// Go code.
package onnxruntime

import (
	"fmt"
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
	return nil
}

// Call this function to cleanup the internal onnxruntime environment when it
// is no longer needed.
func CleanupEnvironment() error {
	var e error
	// TODO: Implement CleanupEnvironment
	// Prior to calling platformCleanup, we need to:
	//  - Destroy the environment
	//  - Destroy any remaining active sessions?

	// platformCleanup primarily unloads the library, so we need to call it
	// last, after unloading the library.
	e = platformCleanup()
	if e != nil {
		return fmt.Errorf("Platform-specific cleanup failed: %w", e)
	}
	return nil
}
