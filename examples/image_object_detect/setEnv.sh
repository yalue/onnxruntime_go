#!/bin/bash
ONNX_LIB_BASE=$(go list -f '{{.Dir}}' -m github.com/8ff/onnxruntime_go)
export CGO_LDFLAGS="-L$ONNX_LIB_BASE/lib/osx -Wl,-rpath,$ONNX_LIB_BASE/lib/osx -lonnxruntime -framework CoreML"
