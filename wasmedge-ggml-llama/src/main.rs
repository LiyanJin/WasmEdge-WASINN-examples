use std::env;
use wasi_nn;

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_name: &str = &args[1];
    let prompt: &str = &args[2];

    let graph =
        wasi_nn::GraphBuilder::new(wasi_nn::GraphEncoding::Ggml, wasi_nn::ExecutionTarget::AUTO)
            .build_from_cache(model_name)
            .expect("Failed to build graph from cache");
    println!("Loaded model into wasi-nn with ID: {:?}", graph);

    let mut context = graph.init_execution_context().expect("Failed to initialize execution context");
    println!("Created wasi-nn execution context with ID: {:?}", context);

    let tensor_data = prompt.as_bytes().to_vec();
    println!("Read input tensor, size in bytes: {}", tensor_data.len());
    context
        .set_input(0, wasi_nn::TensorType::U8, &[1], &tensor_data)
        .expect("Failed to set input");

    // Execute the inference.
    context.compute().expect("Failed to execute model inference");
    println!("Executed model inference");

    // Retrieve the output.
    let mut output_buffer = vec![0u8; 1000];
    context.get_output(0, &mut output_buffer).expect("Failed to retrieve the output");
    let output = String::from_utf8(output_buffer.clone()).expect("Failed to convert output to string");
    println!("Output: {}", output);
}
