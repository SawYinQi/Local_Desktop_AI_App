// Generated Rust types from agent.proto (package "agent").
pub mod agent {
    tonic::include_proto!("agent");
}

use agent::agent_service_client::AgentServiceClient;
use agent::{QueryRequest, VideoRequest};
use serde::Serialize;

const BACKEND_ADDR: &str = "http://localhost:50051";

// serializable struct for sending to frontend.
#[derive(Serialize)]
struct ChatEvent {
    response: String,
    artifact_path: String,
}

// Tell the backend which video this session is about.
#[tauri::command]
async fn upload_video(session_id: String, file_path: String) -> Result<String, String> {
    
    // Open connection to gRPC server
    let mut client = AgentServiceClient::connect(BACKEND_ADDR)
        .await
        .map_err(|e| format!("connect failed: {e}"))?;

    // Call the upload_video RPC and wait for the response
    let resp = client
        .upload_video(VideoRequest { session_id, file_path })
        .await
        .map_err(|e| format!("upload_video failed: {e}"))?
        .into_inner();
    Ok(resp.message)
}

// Send a user query to the backend and stream back the response events.
#[tauri::command]
async fn send_query(session_id: String, query: String) -> Result<Vec<ChatEvent>, String> {
    let mut client = AgentServiceClient::connect(BACKEND_ADDR)
        .await
        .map_err(|e| format!("connect failed: {e}"))?;

    // Call the send_query RPC and get a stream of responses
    let mut stream = client
        .send_query(QueryRequest { session_id, query })
        .await
        .map_err(|e| format!("send_query failed: {e}"))?
        .into_inner();

    // Collect the streamed responses into a vector to send back to the frontend
    let mut out = Vec::new();
    while let Some(msg) = stream.message().await.map_err(|e| format!("stream error: {e}"))? {
        out.push(ChatEvent {
            response: msg.response,
            artifact_path: msg.artifact_path,
        });
    }
    Ok(out)
}

#[tauri::command]
fn open_artifact(path: String) -> Result<(), String> {
    let opener = if cfg!(target_os = "macos") {
        "open"
    } else if cfg!(target_os = "windows") {
        "explorer"
    } else {
        "xdg-open"
    };

    std::process::Command::new(opener)
        .arg(&path)
        .spawn()
        .map_err(|e| format!("Failed to open {path}: {e}"))?;
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![upload_video, send_query, open_artifact])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
