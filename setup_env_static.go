//go:build !windows && static

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

// loadORTHandle resolves the ONNX Runtime symbols that are already linked
// statically into the running executable (the "static" build tag). It does so
// via dlopen(NULL), which returns a handle to the process's own global symbol
// table. The shared environment-setup logic lives in setup_env_common.go and
// calls this function.
//
// There is intentionally no .so fallback path: a main executable cannot be
// dlopen'd by its own file path (glibc rejects it with "cannot dynamically load
// executable"), so dlopen(NULL) is the only supported mechanism once ORT is
// statically linked.
func loadORTHandle() (unsafe.Pointer, error) {
	handle := C.dlopen(nil, C.RTLD_LAZY)
	if handle == nil {
		msg := C.GoString(C.dlerror())
		return nil, fmt.Errorf("Error resolving statically-linked ONNX Runtime via dlopen(NULL): %s", msg)
	}
	return handle, nil
}
