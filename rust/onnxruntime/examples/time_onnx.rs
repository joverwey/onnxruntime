use onnxruntime::session::Session;
use onnxruntime::session_options::SessionOptions;
use onnxruntime::{shaped_data::ShapedData, OnnxError};
use std::{convert::TryInto, path::PathBuf, time::Instant};

fn run(model_path: &str) -> Result<(), OnnxError> {
    let options =
        if cfg!(feature = "gpu") {
            SessionOptions::with_cuda_deafult()?
        }
        else {
            SessionOptions::new()?
        };

    let mut session = Session::from_options(model_path, &options)?;

    let shape = vec![1, 3, 224, 224];
    let input_tensor_size = shape.iter().product();

    let input_tensor_values: Vec<f32> = (0..input_tensor_size)
        .map(|i| i as f32 / (input_tensor_size as f32 + 1.0))
        .collect();

    let shaped_data = ShapedData::new(shape, input_tensor_values)?;

    let inputs = vec![("data_0", session.create_tensor_from_shaped_data(shaped_data)?)];

    let now = Instant::now();
    for i in 0..1000 {
        let outputs = session.run(&inputs)?;

        if i == 0 {
            if let [output] = &outputs[..] {
                let shaped_data: ShapedData<f32> = output.try_into()?;
                for element in shaped_data.elements().iter().take(5) {
                    println!("{}", element)
                }
            } else {
                eprintln!("Expected exactly one tensor returned from the model");
            }
        }
    }
    println!("Finished in {} seconds", now.elapsed().as_secs());

    Ok(())
}

pub fn get_model_path(filename: &str) -> String {
    let mut buf: PathBuf = std::env::current_exe().unwrap();
    while buf.pop() && !buf.ends_with("onnxruntime") {}
    buf.push("csharp");
    buf.push("testdata");
    buf.push(filename);
    buf.to_str().unwrap().into()
}

fn main() {
    let model_path = get_model_path("squeezenet.onnx");
    if let Err(e) = run(&model_path) {
        println!("Unable to create API: {}", e);
    } else {
        println!("Done");
    }
}
