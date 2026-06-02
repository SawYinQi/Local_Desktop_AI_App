fn main() {
    // Generate the Rust gRPC client from agent.proto 
    tonic_build::configure()
        .build_server(false)
        .compile_protos(
            &["../backend/proto/agent.proto"], // the contract
            &["../backend/proto"],             // include dir for imports
        )
        .expect("failed to compile agent.proto");

    tauri_build::build();
}
