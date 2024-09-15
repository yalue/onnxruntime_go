//go:build windows

package onnxruntime_go

// This file includes the Windows-specific code for loading the onnxruntime
// library and setting up the environment.

import (
	"fmt"
	"syscall"
	"unicode/utf16"
	"unicode/utf8"
	"unsafe"
)

// #include "onnxruntime_wrapper.h"
import "C"

// This will contain the handle to the onnxruntime dll if it has been loaded
// successfully.
var libraryHandle syscall.Handle

func platformCleanup() error {
	e := syscall.FreeLibrary(libraryHandle)
	libraryHandle = 0
	return e
}

func platformInitializeEnvironment() error {
	if onnxSharedLibraryPath == "" {
		onnxSharedLibraryPath = "onnxruntime.dll"
	}
	handle, e := syscall.LoadLibrary(onnxSharedLibraryPath)
	if e != nil {
		return fmt.Errorf("Error loading ONNX shared library \"%s\": %w",
			onnxSharedLibraryPath, e)
	}
	getApiBaseProc, e := syscall.GetProcAddress(handle, "OrtGetApiBase")
	if e != nil {
		syscall.FreeLibrary(handle)
		return fmt.Errorf("Error finding OrtGetApiBase function in %s: %w",
			onnxSharedLibraryPath, e)
	}
	ortApiBase, _, e := syscall.SyscallN(uintptr(getApiBaseProc), 0)
	if ortApiBase == 0 {
		syscall.FreeLibrary(handle)
		if e != nil {
			return fmt.Errorf("Error calling OrtGetApiBase: %w", e)
		} else {
			return fmt.Errorf("Error calling OrtGetApiBase")
		}
	}
	tmp := C.SetAPIFromBase((*C.OrtApiBase)(unsafe.Pointer(ortApiBase)))
	if tmp != 0 {
		syscall.FreeLibrary(handle)
		return fmt.Errorf("Error setting ORT API base: %d", tmp)
	}

	// we do not initialize the training API on windows (see setup_env.go)
	// because currently we cannot support the conversion from UTF-8 to wide
	// character. See https://github.com/yalue/onnxruntime_go/pull/56.

	libraryHandle = handle
	return nil
}

// Converts the given string to a UTF-16 string, pointed to by a raw
// *C.char. Note that we actually keep ORTCHAR_T defined to char even
// on Windows, so do _not_ index into this string from Cgo code and expect to
// get correct characters! Instead, this should only be used to obtain pointers
// that are passed to onnxruntime windows DLL functions expecting ORTCHAR_T*
// args. This is required because we undefine _WIN32 for cgo compatibility when
// including onnxruntime_c_api.h, but still interact with a DLL that was
// compiled assuming _WIN32 was defined.
//
// The pointer returned by this function must still be freed using C.free when
// no longer needed. This will return an error if the given string contains
// non-UTF8 characters.
func createOrtCharString(str string) (*C.char, error) {
	src := []uint8(str)
	// Assumed common case: the utf16 buffer contains one uint16 per utf8 byte
	// plus one more for the required null terminator in the C buffer.
	dst := make([]uint16, 0, len(src)+1)
	// Convert UTF-8 to UTF-16 by reading each subsequent rune from src and
	// appending it as UTF-16 to dst.
	for len(src) > 0 {
		r, size := utf8.DecodeRune(src)
		if r == utf8.RuneError {
			return nil, fmt.Errorf("Invalid UTF-8 rune found in \"%s\"", str)
		}
		src = src[size:]
		dst = utf16.AppendRune(dst, r)
	}
	// Make sure dst contains the null terminator. Additionally this will cause
	// us to return an empty string if the original string was empty.
	dst = append(dst, 0)

	// Finally, we need to copy dst into a C array for compatibility with
	// C.CString.
	toReturn := C.calloc(C.size_t(len(dst)), 2)
	if toReturn == nil {
		return nil, fmt.Errorf("Error allocating buffer for the utf16 string")
	}
	C.memcpy(toReturn, unsafe.Pointer(&(dst[0])), C.size_t(len(dst))*2)

	return (*C.char)(toReturn), nil
}
