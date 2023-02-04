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
    "github.com/yalue/onnxruntime"
    "os"
)

func main() {
    // This line may be optional, by default the library will try to load
    // "onnxruntime.dll" on Windows, and "onnxruntime.so" on any other system.
    onnxruntime.SetSharedLibraryPath("path/to/onnxruntime.so")

    err := onnxruntime.InitializeEnvironment()
    defer onnxruntime.DestroyEnvironment()

    // We'll assume that network.onnx takes a single 2x3x4 input tensor and
    // produces a 1x2x3 output tensor.
    session, err := onnxruntime.CreateSimpleSession("path/to/network.onnx",
        onnxruntime.NewShape(2, 3, 4), onnxruntime.NewShape(1, 2, 3))
    defer session.Destroy()

    // Network inputs must be provided as flattened slices of floats. Run() can
    // be called as many times as necessary with a single session.
    err = session.Run([]float32{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7})

    // This will copy the result tensor into a flattened float32 slice.
    outputShape, err := session.OutputShape()
    results := make([]float32, outputShape.FlattenedSize())
    err = session.CopyResults(results)

    // ...
}
```

Full Documentation
------------------

The above example uses a single input and produces a single output, all with
`float32` data.  The `CreateSimpleSession` function supports this, as it is
expected to be a common use case.  However, the library supports far more
options, i.e. using the `CreateSession` function when setting up a session.

The full documentation can be found at [pkg.go.dev](https://pkg.go.dev/github.com/yalue/onnxruntime).

