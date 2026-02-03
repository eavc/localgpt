//! HTTP server for LocalGPT
//!
//! The chat endpoint uses a shared Agent with session persistence across requests.

use anyhow::Result;
use axum::{
    extract::{
        ws::{Message as WsMessage, WebSocket, WebSocketUpgrade},
        Query, State,
    },
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse, Json, Response,
    },
    routing::{get, post},
    Router,
};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};
use tracing::{debug, info};

use crate::agent::{Agent, AgentConfig, StreamEvent};
use crate::config::Config;
use crate::memory::MemoryManager;

pub struct Server {
    config: Config,
}

struct AppState {
    config: Config,
    agent: Arc<Mutex<Agent>>,
}

impl Server {
    pub fn new(config: &Config) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    pub async fn run(&self) -> Result<()> {
        // Create shared agent at startup
        let memory = MemoryManager::new(&self.config.memory)?;
        let agent_config = AgentConfig {
            model: self.config.agent.default_model.clone(),
            context_window: self.config.agent.context_window,
            reserve_tokens: self.config.agent.reserve_tokens,
        };
        let mut agent = Agent::new(agent_config, &self.config, memory).await?;
        agent.new_session().await?;

        let state = Arc::new(AppState {
            config: self.config.clone(),
            agent: Arc::new(Mutex::new(agent)),
        });

        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any);

        let app = Router::new()
            .route("/health", get(health_check))
            .route("/api/chat", post(chat))
            .route("/api/chat/stream", post(chat_stream))
            .route("/api/ws", get(websocket_handler))
            .route("/api/memory/search", get(memory_search))
            .route("/api/memory/stats", get(memory_stats))
            .route("/api/status", get(status))
            .layer(cors)
            .with_state(state);

        let addr: SocketAddr =
            format!("{}:{}", self.config.server.bind, self.config.server.port).parse()?;

        info!("Starting HTTP server on http://{}", addr);

        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }
}

// Error response type
struct AppError(StatusCode, String);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (self.0, self.1).into_response()
    }
}

// Health check endpoint
async fn health_check() -> &'static str {
    "OK"
}

// Status endpoint
#[derive(Serialize)]
struct StatusResponse {
    version: String,
    model: String,
    memory_chunks: usize,
}

async fn status(State(state): State<Arc<AppState>>) -> Json<StatusResponse> {
    let memory = MemoryManager::new(&state.config.memory).ok();

    Json(StatusResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
        model: state.config.agent.default_model.clone(),
        memory_chunks: memory.and_then(|m| m.chunk_count().ok()).unwrap_or(0),
    })
}

// Chat endpoint
#[derive(Deserialize)]
struct ChatRequest {
    message: String,
    #[allow(dead_code)] // TODO: implement model switching per request
    model: Option<String>,
}

#[derive(Serialize)]
struct ChatResponse {
    response: String,
    model: String,
}

async fn chat(State(state): State<Arc<AppState>>, Json(request): Json<ChatRequest>) -> Response {
    let mut agent = state.agent.lock().await;

    // TODO: Handle model change if request.model differs from current

    match agent.chat(&request.message).await {
        Ok(response) => Json(ChatResponse {
            response,
            model: agent.model().to_string(),
        })
        .into_response(),
        Err(e) => AppError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

// Streaming chat endpoint (SSE) with tool support
async fn chat_stream(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatRequest>,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let agent = state.agent.clone();

    let stream = async_stream::stream! {
        let mut agent = agent.lock().await;

        // Use streaming with tools
        match agent.chat_stream_with_tools(&request.message).await {
            Ok(event_stream) => {
                use futures::StreamExt;

                // Pin the stream to iterate over it
                let mut pinned_stream = std::pin::pin!(event_stream);

                while let Some(event) = pinned_stream.next().await {
                    match event {
                        Ok(StreamEvent::Content(content)) => {
                            let data = json!({"type": "content", "delta": content});
                            yield Ok(Event::default().data(data.to_string()));
                        }
                        Ok(StreamEvent::ToolCallStart { name, id }) => {
                            let data = json!({"type": "tool_start", "name": name, "id": id});
                            yield Ok(Event::default().data(data.to_string()));
                        }
                        Ok(StreamEvent::ToolCallEnd { name, id, output }) => {
                            let data = json!({
                                "type": "tool_end",
                                "name": name,
                                "id": id,
                                "output": output.chars().take(500).collect::<String>()
                            });
                            yield Ok(Event::default().data(data.to_string()));
                        }
                        Ok(StreamEvent::Done) => {
                            let data = json!({"type": "done"});
                            yield Ok(Event::default().data(data.to_string()));
                        }
                        Err(e) => {
                            yield Ok(Event::default().data(json!({"error": e.to_string()}).to_string()));
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                yield Ok(Event::default().data(json!({"error": e.to_string()}).to_string()));
            }
        }

        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(stream)
}

// Memory search endpoint
#[derive(Deserialize)]
struct SearchQuery {
    q: String,
    limit: Option<usize>,
}

#[derive(Serialize)]
struct SearchResult {
    file: String,
    line_start: i32,
    line_end: i32,
    content: String,
    score: f64,
}

#[derive(Serialize)]
struct SearchResponse {
    results: Vec<SearchResult>,
    query: String,
}

async fn memory_search(
    State(state): State<Arc<AppState>>,
    Query(query): Query<SearchQuery>,
) -> Response {
    match memory_search_inner(&state.config.memory, &query.q, query.limit) {
        Ok(response) => Json(response).into_response(),
        Err(e) => AppError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

fn memory_search_inner(
    config: &crate::config::MemoryConfig,
    query: &str,
    limit: Option<usize>,
) -> Result<SearchResponse, anyhow::Error> {
    let memory = MemoryManager::new(config)?;

    let limit = limit.unwrap_or(10);
    let results = memory.search(query, limit)?;

    let results: Vec<SearchResult> = results
        .into_iter()
        .map(|r| SearchResult {
            file: r.file,
            line_start: r.line_start,
            line_end: r.line_end,
            content: r.content,
            score: r.score,
        })
        .collect();

    Ok(SearchResponse {
        results,
        query: query.to_string(),
    })
}

// Memory stats endpoint
#[derive(Serialize)]
struct StatsResponse {
    workspace: String,
    total_files: usize,
    total_chunks: usize,
    index_size_kb: u64,
}

async fn memory_stats(State(state): State<Arc<AppState>>) -> Response {
    match memory_stats_inner(&state.config.memory) {
        Ok(response) => Json(response).into_response(),
        Err(e) => AppError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

fn memory_stats_inner(
    config: &crate::config::MemoryConfig,
) -> Result<StatsResponse, anyhow::Error> {
    let memory = MemoryManager::new(config)?;
    let stats = memory.stats()?;

    Ok(StatsResponse {
        workspace: stats.workspace,
        total_files: stats.total_files,
        total_chunks: stats.total_chunks,
        index_size_kb: stats.index_size_kb,
    })
}

// WebSocket handler
async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_websocket(socket, state))
}

/// WebSocket message types
#[derive(Deserialize)]
#[serde(tag = "type")]
enum WsIncoming {
    /// Chat message
    #[serde(rename = "chat")]
    Chat { message: String },
    /// Ping for keepalive
    #[serde(rename = "ping")]
    Ping,
}

#[derive(Serialize)]
#[serde(tag = "type")]
#[allow(dead_code)] // ToolStart/ToolEnd reserved for streaming with tools
enum WsOutgoing {
    /// Connection established
    #[serde(rename = "connected")]
    Connected { session_id: String },
    /// Text content chunk
    #[serde(rename = "content")]
    Content { delta: String },
    /// Tool call started
    #[serde(rename = "tool_start")]
    ToolStart { name: String, id: String },
    /// Tool call completed
    #[serde(rename = "tool_end")]
    ToolEnd { name: String, id: String, output: String },
    /// Message complete
    #[serde(rename = "done")]
    Done,
    /// Pong response
    #[serde(rename = "pong")]
    Pong,
    /// Error
    #[serde(rename = "error")]
    Error { message: String },
}

async fn handle_websocket(socket: WebSocket, state: Arc<AppState>) {
    let (mut sender, mut receiver) = socket.split();

    // Send connected message
    let session_id = {
        let agent = state.agent.lock().await;
        agent.session_status().id
    };

    let connected = WsOutgoing::Connected { session_id };
    if let Ok(json) = serde_json::to_string(&connected) {
        let _ = sender.send(WsMessage::Text(json.into())).await;
    }

    debug!("WebSocket client connected");

    // Process incoming messages
    while let Some(msg) = receiver.next().await {
        match msg {
            Ok(WsMessage::Text(text)) => {
                // Parse incoming message
                match serde_json::from_str::<WsIncoming>(&text) {
                    Ok(WsIncoming::Chat { message }) => {
                        debug!("WebSocket chat: {}", message);

                        // Process chat (with tool support)
                        let mut agent = state.agent.lock().await;

                        match agent.chat(&message).await {
                            Ok(response) => {
                                // Send response as content
                                let content = WsOutgoing::Content {
                                    delta: response,
                                };
                                if let Ok(json) = serde_json::to_string(&content) {
                                    let _ = sender.send(WsMessage::Text(json.into())).await;
                                }

                                // Send done
                                let done = WsOutgoing::Done;
                                if let Ok(json) = serde_json::to_string(&done) {
                                    let _ = sender.send(WsMessage::Text(json.into())).await;
                                }
                            }
                            Err(e) => {
                                let error = WsOutgoing::Error {
                                    message: e.to_string(),
                                };
                                if let Ok(json) = serde_json::to_string(&error) {
                                    let _ = sender.send(WsMessage::Text(json.into())).await;
                                }
                            }
                        }
                    }
                    Ok(WsIncoming::Ping) => {
                        let pong = WsOutgoing::Pong;
                        if let Ok(json) = serde_json::to_string(&pong) {
                            let _ = sender.send(WsMessage::Text(json.into())).await;
                        }
                    }
                    Err(e) => {
                        let error = WsOutgoing::Error {
                            message: format!("Invalid message format: {}", e),
                        };
                        if let Ok(json) = serde_json::to_string(&error) {
                            let _ = sender.send(WsMessage::Text(json.into())).await;
                        }
                    }
                }
            }
            Ok(WsMessage::Ping(data)) => {
                let _ = sender.send(WsMessage::Pong(data)).await;
            }
            Ok(WsMessage::Close(_)) => {
                debug!("WebSocket client disconnected");
                break;
            }
            Err(e) => {
                debug!("WebSocket error: {}", e);
                break;
            }
            _ => {}
        }
    }

    debug!("WebSocket connection closed");
}
