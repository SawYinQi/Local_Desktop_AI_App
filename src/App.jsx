import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";

function App() {
  // Load saved chat on startup so history survives app restarts.
  const [messages, setMessages] = useState(() => {
    const saved = localStorage.getItem("chatHistory"); 
    return saved ? JSON.parse(saved) : [];
  });
  // Remember the chosen video path so it survives app restarts too.
  const [videoPath, setVideoPath] = useState(() => localStorage.getItem("videoPath") || "");
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [dots, setDots] = useState(""); 

  // Persist chat history on every change.
  useEffect(() => {
    localStorage.setItem("chatHistory", JSON.stringify(messages));
  }, [messages]);

  // Persist the selected video path.
  useEffect(() => {
    localStorage.setItem("videoPath", videoPath);
  }, [videoPath]);

  // cycle dot animation when busy
  useEffect(() => {
    if (!busy) {
      setDots("");
        return;
    }
    const interval = setInterval(() => {
        setDots((prev) => (prev.length >= 3 ? "" : prev + "."));
    }, 400);

    return () => clearInterval(interval);
  }, [busy]);

  // Open a native file picker to choose the video this session is about.
  async function chooseVideo() {
    const selected = await open({
      multiple: false,
      filters: [{ name: "Video", extensions: ["mp4", "mov", "m4v", "webm"] }],
    });
    if (typeof selected === "string") setVideoPath(selected);
  }

  // Send the user's query to the backend, along with the video path if relevant. Stream responses back and update the chat.
  async function send() {
    const query = input.trim();

    if (!query || busy) return;

    setInput("");
    setMessages((m) => [...m, { role: "user", text: query }]); 
    setBusy(true);

    const sessionId = videoPath ? `video:${videoPath}` : "general";

    // try query, await responses, catch errors, and always unset busy at the end
    try {
      await invoke("upload_video", { sessionId, filePath: videoPath });
      const responses = await invoke("send_query", { sessionId, query }); 

      responses.forEach((event) =>
        setMessages((m) => [...m, {
          role: "assistant",
          text: event.response,
          artifact_path: event.artifact_path,
        }])
      );

    } catch (e) {
      setMessages((m) => [...m, { role: "error", text: String(e) }]);
    } finally {
      setBusy(false);
    }
  }


  const videoName = videoPath ? videoPath.split(/[\\/]/).pop() : null;

  return (
    <main style={{ maxWidth: 700, margin: "0 auto", padding: 16, fontFamily: "system-ui" }}>

      <h2>Local Desktop AI</h2>

      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>

        <button onClick={ () => { !videoPath? chooseVideo() : setVideoPath("") }} 
        style={{
          padding: "5px 12px",
          background: videoPath ? "#d33" : "#2c7a4b",
          color: "white",
          border: "none",
          cursor: "pointer",
          }}>
          {videoPath ? "Clear video" : "Choose video"}</button>

        <span style={{ color: "#6b6b6b", fontSize: 13 , background: "#eee", padding: "4px 6px"}}>
          {videoName ? `video: ${videoName}` : "No video selected"}
        </span>
      </div>

      <div style={{ border: "1px solid #ccc", borderRadius: 8, height: 380, overflowY: "auto", padding: 12, marginBottom: 12 }}>
        {messages.length === 0 && (
          <div style={{ color: "#aaa" }}>Ask something, e.g. "transcribe the video"</div>
        )}
        {messages.map((m, i) => (
          <div key={i} style={{ margin: "8px 0", textAlign: m.role === "user" ? "right" : "left" }}>
            <div
              style={{
                display: "inline-block", padding: "6px 10px", borderRadius: 8, maxWidth: "85%",
                whiteSpace: "pre-wrap", textAlign: "left",
                background: m.role === "user" ? "#0b7d4f"
                  : m.role === "error" ? "#d33"
                  : "#eee",
                color: m.role === "assistant" ? "#111" : "#fff",
              }}
            >
              <div>{m.text}</div>
              {m.artifact_path && (
                <button
                  onClick={() => invoke("open_artifact", { path: m.artifact_path })}
                  style={{
                    display: "inline-block",
                    marginTop: 8,
                    padding: "5px 10px",
                    background: "#2c7a4b",
                    color: "white",
                    border: "none",
                    borderRadius: 5,
                    cursor: "pointer",
                    fontSize: 12,
                    fontWeight: 500,
                  }}
                >
                  Open {m.artifact_path.endsWith(".pdf") ? "PDF" : "PPTX"}
                </button>
              )}
            </div>
          </div>
        ))}

        {busy && <div style={{
          display: "inline-block", padding: "6px 10px", 
          borderRadius: 8, color: "#888" ,
          background: "#eee", padding: "4px 6px"}}><i>Thinking{dots}</i></div>}
          
      </div>
      <form onSubmit={(e) => { e.preventDefault(); send(); }} style={{ display: "flex", gap: 8 }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about the video…"
          style={{ flex: 1, padding: 8, borderRadius: 6, border: "1px solid #ccc" }}
        />
        <button type="submit" disabled={busy} style={{ padding: "8px 16px" }}>Send</button>
      </form>
    </main>
  );
}

export default App;
