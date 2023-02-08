package onnxruntime_go

// This file contains definitions for the generic tensor data types we support.

// #include "onnxruntime_wrapper.h"
import "C"

import (
	"reflect"
)

type FloatData interface {
	~float32 | ~float64
}

type IntData interface {
	~int8 | ~uint8 | ~int16 | ~uint16 | ~int32 | ~uint32 | ~int64 | ~uint64
}

// This is used as a type constraint for the generic Tensor type.
type TensorData interface {
	FloatData | IntData
}

// Returns the ONNX enum value used to indicate TensorData type T.
func GetTensorElementDataType[T TensorData]() C.ONNXTensorElementDataType {
	// Sadly, we can't do type assertions to get underlying types, so we need
	// to use reflect here instead.
	var v T
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
	}
	return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
}
