package onnxruntime_go

// This file contains definitions for the generic tensor data types we support.

// #include "onnxruntime_wrapper.h"
import "C"

import (
	"reflect"

	"github.com/x448/float16"
)

type FloatData interface {
	~float32 | ~float64 | float16.Float16
}

type IntData interface {
	~int8 | ~uint8 | ~int16 | ~uint16 | ~int32 | ~uint32 | ~int64 | ~uint64
}

// This is used as a type constraint for the generic Tensor type.
type TensorData interface {
	FloatData | IntData | ~bool
}

// Returns the ONNX enum value used to indicate TensorData type T.
func GetTensorElementDataType[T TensorData]() C.ONNXTensorElementDataType {
	// Sadly, we can't do type assertions to get underlying types, so we need
	// to use reflect here instead.
	var v T
	// There MUST be a better way......
	// but since float16 is based on int, this is it...
	if reflect.ValueOf(v).Type().Name() == "Float16" {
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
	}
	kind := reflect.ValueOf(v).Kind()
	switch kind {
	case reflect.Float64:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
	case reflect.Float32:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
	case reflect.Int8:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
	case reflect.Uint8:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
	case reflect.Int16:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
	case reflect.Uint16:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
	case reflect.Int32:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
	case reflect.Uint32:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
	case reflect.Int64:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
	case reflect.Uint64:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64
	case reflect.Bool:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
	}
	return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
}
