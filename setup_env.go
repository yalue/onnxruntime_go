//go:build !windows && !static

package onnxruntime_go

/*
#include <dlfcn.h>
#include "onnxruntime_wrapper.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// loadORTHandle loads the onnxruntime shared library from disk (the default
// build). The shared environment-setup logic lives in setup_env_common.go and
// calls this function; only the "how to obtain the handle" step differs
// between the default and "static" build tags.
func loadORTHandle() (unsafe.Pointer, error) {
	if onnxSharedLibraryPath == "" {
		onnxSharedLibraryPath = "onnxruntime.so"
	}
	cName := C.CString(onnxSharedLibraryPath)
	defer C.free(unsafe.Pointer(cName))
	handle := C.dlopen(cName, C.RTLD_LAZY)
	if handle == nil {
		msg := C.GoString(C.dlerror())
		return nil, fmt.Errorf("Error loading ONNX shared library \"%s\": %s",
			onnxSharedLibraryPath, msg)
	}
	return handle, nil
}
