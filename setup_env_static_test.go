//go:build !windows && static

package onnxruntime_go

import "testing"

// TestStaticRuntimeInitialization verifies that, when ONNX Runtime is linked
// statically into the test binary, the "static" build tag resolves
// OrtGetApiBase from the process's own symbol table via dlopen(NULL) and the
// environment initializes successfully.
//
// To actually exercise this path (instead of skipping), build the test binary
// with ORT statically linked and its symbols exported, e.g.:
//
//	CGO_LDFLAGS="-Wl,--whole-archive -L<ORT_LIB_DIR> -lonnxruntime \
//	  -Wl,--no-whole-archive -Wl,--export-dynamic -lstdc++" \
//	  go test -tags static -run TestStaticRuntimeInitialization ./...
//
// If ORT is NOT statically linked (the default for a plain `go test -tags
// static`), dlopen(NULL) won't expose OrtGetApiBase and we skip rather than
// fail: the loader still runs without panicking, which is what we can prove in
// every environment.
func TestStaticRuntimeInitialization(t *testing.T) {
	if IsInitialized() {
		return
	}
	err := InitializeEnvironment()
	if err != nil {
		t.Skipf("ORT not statically linked into this binary; "+
			"dlopen(NULL) could not resolve OrtGetApiBase: %v", err)
	}
	defer DestroyEnvironment()
	if !IsInitialized() {
		t.Fatal("InitializeEnvironment succeeded but IsInitialized() is false")
	}
}

// TestStaticSetSharedLibraryPathIgnored verifies that, under the static build,
// the path passed to SetSharedLibraryPath is ignored: initialization must still
// succeed (or skip) regardless of the value previously configured.
func TestStaticSetSharedLibraryPathIgnored(t *testing.T) {
	SetSharedLibraryPath("/nonexistent/path/onnxruntime.so")
	if IsInitialized() {
		return
	}
	err := InitializeEnvironment()
	if err != nil {
		t.Skipf("ORT not statically linked into this binary (path ignored): %v", err)
	}
	defer DestroyEnvironment()
	if !IsInitialized() {
		t.Fatal("InitializeEnvironment succeeded but IsInitialized() is false")
	}
}
