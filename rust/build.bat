cargo build
copy ..\build\Windows\RelWithDebInfo\RelWithDebInfo\onnxruntime.dll target\debug\examples\onnxruntime.dll /Y
copy ..\build\Windows\RelWithDebInfo\RelWithDebInfo\onnxruntime.dll target\debug\deps\onnxruntime.dll /Y
cargo test
cargo run --example time_onnx

cargo build --release
copy ..\build\Windows\RelWithDebInfo\RelWithDebInfo\onnxruntime.dll target\release\examples\onnxruntime.dll /Y
copy ..\build\Windows\RelWithDebInfo\RelWithDebInfo\onnxruntime.dll target\release\deps\onnxruntime.dll /Y
cargo test --release
cargo run --release --example time_onnx