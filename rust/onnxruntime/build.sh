
cargo build
cp ../../build/Linux/Debug/libonnxruntime.so target/debug/examples/libonnxruntime.so.1.4.0
cp ../../build/Linux/Debug/libonnxruntime.so target/debug/deps/libonnxruntime.so.1.4.0
cargo test
cargo build --release
cargo run --example time_onnx

cp ../../build/Linux/Release/libonnxruntime.so target/release/examples/libonnxruntime.so.1.4.0
cp ../../build/Linux/Release/libonnxruntime.so target/release/deps/libonnxruntime.so.1.4.0
cargo test --release
cargo run --release --example time_onnx
