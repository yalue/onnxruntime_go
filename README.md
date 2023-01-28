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
tensor, both of which contain 32-bit floating point values.

```
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
    if err != nil {
        fmt.Printf("Failed initializing onnxruntime: %s\n", err)
        os.Exit(1)
    }
    defer onnxruntime.CleanupEnvironment()

    // We'll assume that network.onnx takes a single 2x3x4 input tensor and
    // produces a 1x2x2 output tensor.
    inputShape := []int64{1, 2, 3}
    outputShape := []int64{1, 2, 2}
    session, err := onnxruntime.CreateSimpleSession("path/to/network.onnx",
        inputShape, outputShape)
    if err != nil {
        fmt.Printf("Error creating session: %s\n", err)
        os.Exit(1)
    }
    defer session.Destroy()

    // Network inputs must be provided as flattened slices of floats. Run() can
    // be called as many times as necessary with a single session.
    err := session.Run([]float32{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6})
    if err != nil {
        fmt.Printf("Error running the network: %s\n", err)
        os.Exit(1)
    }

    // This will be a flattened slice containing the elements in the 1x2x2
    // output tensor.
    results := session.Results()


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

