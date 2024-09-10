Contribution Guidelines
=======================

This library began as a personal project, and is primarily still maintained as
such.  The following list of guidelines is not necessarily exhaustive, and,
ultimately, any contribution is subject to the maintainer's discretion. That
being said, contributions are welcome, and most recent new features have been
added by users who need them!

Coding Style
------------

 - Go code must be formatted using the official `gofmt` tool.

 - C code should adhere to the portions of Google's C++ style guide that
   apply to C.

 - If at all possible, any Go or C code should have at most 80 character lines.
   (This may not be enforced very strictly.)

 - Purely stylistic changes are unlikely to be accepted. Instead, the
   maintainer or other contributers may make small stylistic adjustments to
   surrounding code as part of other contributions.

 - Attempt to mimic the existing style of the surrounding code.


Documentation
-------------

 - All Go types, public-facing functions, and nontrivial internal functions
   must include a comment on their intended usage, to be parsed by godoc.

 - As per the google C++ style guide, all C functions must be documented with a
   comment as well.  If a C function is defined in a header file, the comment
   should appear with the definition in the header. If it's a static function
   in a `.c` file, the comment should appear with the function definition.


Tests
-----

 - All new features and bugfixes must include a basic unit test (in
   `onnxruntime_test.go` or `onnxruntime_training_test.go`) to serve as a
   sanity check.

 - If a test is for a platform-dependent or execution-provider-dependent
   feature, the test must be skipped if run on an unsupported system.

 - No tests should panic.  Always check errors and fail rather than allowing
   tests to panic.

 - Every change must ensure that `go test -v -bench=.` passes on every
   supported platform.

 - Every test failure should be accompanied by a message containing the reason,
   either using `t.Logf()`, `t.Errorf()`, or `t.Fatalf()`.


Adding New Files
----------------

 - Apart from testing data, try not to add new source files.

 - Do not add third-party code or headers.  The only exceptions for now are
   `onnxruntime_c_api.h` and `onnxruntime_training_c_api.h`.

 - No C++ at all. Developing Go-to-C wrappers is annoying enough as it is.

 - Do not add any new `onnxruntime` shared libraries under `test_data`. I know
   there are additional platforms that would be nice to include (such as
   `x86_64` Linux), but I do not want this project turning into an unofficial
   distribution channel for onnxruntime libraries.  It also clogs up the git
   repo with large files, and increases the size of the history every time
   these files are updated.  The libraries that are included were only intended
   to allow a majority of users to run `go test -v -bench=.` without further
   setup or modification. Currently: amd64 Windows, arm64 Linux (I wish I
   hadn't included this!), arm64 osx, and amd64 osx. All other users must set
   the `ONNXRUNTIME_SHARED_LIBRARY_PATH` environment variable to a valid path
   to the correct `onnxruntime` shared library file prior to running tests.

 - If you need to add a .onnx file for a test, place both the .onnx file
   _and_ the script used to generate it into `test_data/`.

 - Keep any testing .onnx files as small as possible.

 - Without a good reason (i.e., implementing an entire class of APIs such as
   training), avoid adding new Go files---just add to `onnxruntime_go.go`.


Dependencies
------------

 - Avoid Go or C dependencies outside of the language's standard libraries.
   This package currently does not depend on any third-party Go modules, and
   it would be great to keep it this way.

 - Python scripts within `test_data/` can use whatever dependencies they need,
   because end users should not be required to run the python files, and the
   `.onnx` file they produce should already be included.


C-Specific Stuff
----------------

 - Minimize Go management of C-allocated memory as much as possible. For
   example, see the `convertORTString` function on `onnxruntime_go.go`, which
   copies a C-allocated string into a garbage-collected go `string`.

 - If you need to use a `OrtAllocator` in onnxruntime's C API, always use the
   default `OrtAllocator` returned by
   `ort_api->GetAllocatorWithDefaultOptions()`.

 - ONNXRuntime APIs requiring file paths typically use `ORTCHAR_T*`
   strings. On Linux/OSX/etc, these should be UTF-8, but on Windows they will
   be wide-character strings. (Our tricks with `#include` to make them look
   like `char*` to C code even on Windows, but the DLL still expects a
   `wchar_t*`.)  The important takeaway: when passing `ORTCHAR_T*`
   values to the onnxruntime C API, use the `createOrtCharString(...)`
   function. It converts a Go string to a C string, but unlike `C.CString`, it
   will do UTF8 to UTF16 conversion on Windows. (On Linux, it simply wraps
   `C.CString`.)


A Few Notes on Organization
---------------------------

 - The `onnxruntime` C API uses a struct containing function pointers. Cgo
   can't directly invoke functions via pointers, so `onnxruntime_wrapper.c`
   (along with the associated header file) are used to provide top-level C
   functions that call the function pointers within the `OrtApi` struct.

 - Linux and OSX use `dlopen` to load the onnxruntime shared library, but this
   isn't possible on Windows, which instead can use the `syscall.LoadLibrary()`
   function from Go's standard library. This different behavior is locked
   behind build constraints in `setup_env.go` and `setup_env_windows.go`,
   respectively.

