Cross-Platform `onnxruntime` Wrapper for Go
===========================================

About
-----

This library seeks to provide an interface for loading and executing neural
networks from Go(lang) code, while remaining as simple to use as possible.

A few example applications using this library can be found in the
[`onnxruntime_go_examples` repository](https://github.com/yalue/onnxruntime_go_examples).

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

**IMPORTANT:** As of onnxruntime_go v1.12.0 or above, for CUDA acceleration we
now require the use of CUDA 12.x and CuDNN 9.x (as required by onnxruntime
v1.19.0+). Those wishing to stay on CUDA 11.8 should remain on onnxruntime_go
v1.11.0 or below.

Note on onnxruntime Library Versions
------------------------------------

At the time of writing, this library uses version 1.19.0 of the onnxruntime
C API headers.  So, it will probably only work with version 1.19.0 of the
onnxruntime shared libraries, as well.  If you need to use a different version,
or if I get behind on updating this repository, updating or changing the
onnxruntime version should be fairly easy:

 1. Replace the `onnxruntime_c_api.h` file with the version corresponding to
    the onnxruntime version you wish to use.

 2. Replace the `test_data/onnxruntime.dll` (or `test_data/onnxruntime*.so`)
    file with the version corresponding to the onnxruntime version you wish to
    use.

 3. (If you care about DirectML support) Verify that the entries in the
    `DummyOrtDMLAPI` struct in `onnxruntime_wrapper.c` match the order in which
    they appear in the `OrtDmlApi` struct from the `dml_provider_factory.h`
    header in the official repo.  See the comment on this struct in
    `onnxruntime_wrapper.c` for more information.

Note that both the C API header and the shared library files are available to
download from the releases page in the
[official repo](https://github.com/microsoft/onnxruntime). Download the archive
for the release you want to use, and extract it. The header file is located in
the "include" subdirectory, and the shared library will be located in the "lib"
subdirectory. (On Linux systems, you'll need the version of the .so with the
appended version numbers, e.g., `libonnxruntime.so.1.19.0`, and _not_ the
`libonnxruntime.so`, which is just a symbolic link.)  The archive will contain
several other files containing C++ headers, debug symbols, and so on, but you
shouldn't need anything other than the single onnxruntime shared library and
`onnxruntime_c_api.h`.  (The exception is if you're wanting to enable GPU
support, where you may need other shared-library files, such as
`execution_providers_cuda.dll` and `execution_providers_shared.dll` on Windows.)


Requirements
------------

To use this library, you'll need a version of Go with cgo support.  If you are
not using an amd64 version of Windows or Linux (or if you want to provide your
own library for some other reason), you simply need to provide the correct path
to the shared library when initializing the wrapper.  This is seen in the first
few lines of the following example.

Note that if you want to use CUDA, you'll need to be using a version of the
onnxruntime shared library with CUDA support, as well as be using a CUDA
version supported by the underlying version of your onnxruntime library. For
example, version 1.19.0 of the onnxruntime library only supports CUDA versions
12.x. See
[the onnxruntime CUDA support documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
for more specifics.


Example Usage
-------------

The full documentation can be found at [pkg.go.dev](https://pkg.go.dev/github.com/yalue/onnxruntime_go).

Additionally, several example command-line applications complete with necessary
networks and data can be found in the
[`onnxruntime_go_examples` repository](https://github.com/yalue/onnxruntime_go_examples).

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
    // This line _may_ be optional; by default the library will try to load
    // "onnxruntime.dll" on Windows, and "onnxruntime.so" on any other system.
    // For stability, it is probably a good idea to always set this explicitly.
    ort.SetSharedLibraryPath("path/to/onnxruntime.so")

    err := ort.InitializeEnvironment()
    if err != nil {
        panic(err)
    }
    defer ort.DestroyEnvironment()

    // For a slight performance boost and convenience when re-using existing
    // tensors, this library expects the user to create all input and output
    // tensors prior to creating the session. If this isn't ideal for your use
    // case, see the DynamicAdvancedSession type in the documnentation, which
    // allows input and output tensors to be specified when calling Run()
    // rather than when initializing a session.
    inputData := []float32{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
    inputShape := ort.NewShape(2, 5)
    inputTensor, err := ort.NewTensor(inputShape, inputData)
    defer inputTensor.Destroy()
    // This hypothetical network maps a 2x5 input -> 2x3x4 output.
    outputShape := ort.NewShape(2, 3, 4)
    outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
    defer outputTensor.Destroy()

    session, err := ort.NewAdvancedSession("path/to/network.onnx",
        []string{"Input 1 Name"}, []string{"Output 1 Name"},
        []ort.Value{inputTensor}, []ort.Value{outputTensor}, nil)
    defer session.Destroy()

    // Calling Run() will run the network, reading the current contents of the
    // input tensors and modifying the contents of the output tensors.
    err = session.Run()

    // Get a slice view of the output tensor's data.
    outputData := outputTensor.GetData()

    // If you want to run the network on a different input, all you need to do
    // is modify the input tensor data (available via inputTensor.GetData())
    // and call Run() again.

    // ...
}
```


Deprecated APIs
---------------

Older versions of this library used a typed `Session[T]` struct to keep track
of sessions. In retrospect, associating type parameters with Sessions was
unnecessary, and the `AdvancedSession` type, along with its associated APIs,
was added to rectify this mistake.  For backwards compatibility, the old typed
`Session[T]` and `DynamicSession[T]` types are still included and unlikely to
be removed.  However, they now delegate their functionality to
`AdvancedSession` internally.  New code should always favor using
`AdvancedSession` directly.


Running Tests and System Compatibility for Testing
--------------------------------------------------

Navigate to this directory and run `go test -v`, or optionally
`go test -v -bench=.`.  All tests should pass; tests relating to CUDA or other
accelerator support will be skipped on systems or onnxruntime builds that don't
support them.

Currently, this repository includes a copy of the onnxruntime shared libraries
for a few systems, including AMD64 windows, ARM64 Linux, and ARM64 darwin.
These should allow tests to pass on those systems without users needing to copy
additional libraries beyond cloning this repository. In the future, however,
this may change if support for more systems are added or removed.

You may want to use a different version of the `onnxruntime` shared library for
a couple reasons.  In particular:

 1. The included shared library copies do not include support for CUDA or other
    accelerated execution providers, so CUDA-related tests will always be
    skipped if you use the default libraries in this repo.

 2. Many systems, including AMD64 and i386 Linux, and x86 osx, do not currently
    have shared libraries included in `test_data/` in the first place. (I would
    like to keep this directory, and the overall repo, smaller by keeping the
    number of shared libraries small.)

If these or other reasons apply to you, the test code will check the
`ONNXRUNTIME_SHARED_LIBRARY_PATH` environment variable before attempting to
load a library from `test_data/`. So, if you are using one of these systems or
want accelerator-related tests to run, you should set the environment variable
to the path to the onnxruntime shared library.  Afterwards, `go test -v` should
run and pass.


Training API Support
--------------------

This wrapper supports the onnxruntime training API on limited platforms. See
the `NewTrainingSession` and associated data types or functions to use it. So
far, the training API has only been tested on Linux, on `x86_64` architectures.

If you are not sure whether your platform or build of onnxruntime supports
training, you can call `onnxruntime_go.IsTrainingSupported()`, which will
return `true` if training is supported on your system.

