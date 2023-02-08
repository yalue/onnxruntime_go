Cross-Platform `onnxruntime` Wrapper for Go
===========================================

About
-----

This library seeks to provide an interface for loading and executing neural
networks from Go(lang) code, while remaining as simple to use as possible.

The [onnxruntime](https://github.com/microsoft/onnxruntime) library provides a
way to load and execute ONNX-format neural networks, though the library
primarily supports C and C++ APIs.  Several efforts exist to have written
Go(lang) wrappers for the `onnxruntime` library, but as far as I can tell, none
of these existing Go wrappers support Windows. This is due to the fact that
Microsoft's `onnxruntime` library assumes the user will be using the MSVC
compiler on Windows systems, while CGo on Windows requires using Mingw.

This wrapper works around the issues by manually loading the `onnxruntime`
shared library, removing any dependency on the `onnxruntime` source code beyond
the header files.  Naturally, this approach works equally well on non-Windows
systems.

Additionally, this library uses Go's recent addition of generics to support
multiple Tensor data types; see the `NewTensor` or `NewEmptyTensor` functions.

Requirements
------------

To use this library, you'll need a version of Go with cgo support.  If you are
not using an amd64 version of Windows or Linux (or if you want to provide your
own library for some other reason), you simply need to provide the correct path
to the shared library when initializing the wrapper.  This is seen in the first
few lines of the following example.


Example Usage
-------------

The following example illustrates how this library can be used to load and run
an ONNX network taking a single input tensor and producing a single output
tensor, both of which contain 32-bit floating point values.  Note that error
handling is omitted; each of the functions returns an err value, which will be
non-nil in the case of failure.

```go
import (
    "fmt"
    ort "github.com/yalue/onnxruntime_go"
    "os"
)

func main() {
    // This line may be optional, by default the library will try to load
    // "onnxruntime.dll" on Windows, and "onnxruntime.so" on any other system.
    ort.SetSharedLibraryPath("path/to/onnxruntime.so")

    err := ort.InitializeEnvironment()
    defer ort.DestroyEnvironment()

    // To make it easier to work with the C API, this library requires the user
    // to create all input and output tensors prior to creating the session.
    inputData := []float32{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
    inputShape := ort.Shape([]int64{2, 5})
    inputTensor, err := ort.NewTensor(inputShape, inputData)
    defer inputTensor.Destroy()
    // This hypothetical network maps a 2x5 input -> 2x3x4 output.
    outputShape := ort.Shape([]int64{2, 3, 4})
    outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
    defer outputTensor.Destroy()

    session, err := ort.NewSession[float32]("path/to/network.onnx",
        []string{"Input 1 Name"}, []string{"Output 1 Name"},
        []*Tensor[float32]{inputTensor}, []*Tensor[float32]{outputTensor})
    defer session.Destroy()

    // Calling Run() will run the network, reading the current contents of the
    // input tensors and modifying the contents of the output tensors. Simply
    // modify the input tensor's data (available via inputTensor.GetData())
    // before calling Run().
    err = session.Run()

    outputData := outputTensor.GetData()

    // ...
}
```

The full documentation can be found at [pkg.go.dev](https://pkg.go.dev/github.com/yalue/onnxruntime_go).

