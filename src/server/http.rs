//! HTTP server for LocalGPT
//!
//! Supports multiple sessions with session ID-based routing.
//! Sessions are created on demand and cached for reuse.

use anyhow::Result;
use axum::{
    extract::{
        ws::{Message as WsMessage, WebSocket, WebSocketUpgrade},
        Path, Query, State,
    },
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse, Json, Response,
    },
    routing::{delete, get, post},
    Router,
};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};
use tracing::{debug, info};

use crate::agent::{Agent, AgentConfig, StreamEvent};
use crate::config::Config;
use crate::memory::MemoryManager;

/// Session timeout (30 minutes of inactivity)
const SESSION_TIMEOUT: Duration = Duration::from_secs(30 * 60);

/// Maximum number of concurrent sessions
const MAX_SESSIONS: usize = 100;

pub struct Server {
    config: Config,
}

struct SessionEntry {
    agent: Agent,
    last_accessed: Instant,
}

struct AppState {
    config: Config,
    sessions: Mutex<HashMap<String, SessionEntry>>,
}

impl Server {
    pub fn new(config: &Config) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    pub async fn run(&self) -> Result<()> {
        let state = Arc::new(AppState {
            config: self.config.clone(),
            sessions: Mutex::new(HashMap::new()),
        });

        // Spawn session cleanup task
        let cleanup_state = state.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            loop {
                interval.tick().await;
                cleanup_expired_sessions(&cleanup_state).await;
            }
        });

        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any);

        let app = Router::new()
            .route("/health", get(health_check))
            .route("/api/sessions", post(create_session))
            .route("/api/sessions", get(list_sessions))
            .route("/api/sessions/{session_id}", delete(delete_session))
            .route("/api/sessions/{session_id}", get(get_session_status))
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

// Session cleanup task
async fn cleanup_expired_sessions(state: &Arc<AppState>) {
    let mut sessions = state.sessions.lock().await;
    let before_count = sessions.len();

    sessions.retain(|id, entry| {
        let expired = entry.last_accessed.elapsed() > SESSION_TIMEOUT;
        if expired {
            debug!("Expiring session: {}", id);
        }
        !expired
    });

    let removed = before_count - sessions.len();
    if removed > 0 {
        info!("Cleaned up {} expired sessions", removed);
    }
}

// Get or create a session
async fn get_or_create_session(
    state: &Arc<AppState>,
    session_id: Option<String>,
) -> Result<String, AppError> {
    let mut sessions = state.sessions.lock().await;

    // If session_id provided, try to use existing session
    if let Some(ref id) = session_id {
        if sessions.contains_key(id) {
            // Update last accessed time
            if let Some(entry) = sessions.get_mut(id) {
                entry.last_accessed = Instant::now();
            }
            return Ok(id.clone());
        }
    }

    // Check session limit
    if sessions.len() >= MAX_SESSIONS {
        // Try to remove oldest session
        if let Some(oldest_id) = sessions
            .iter()
            .min_by_key(|(_, e)| e.last_accessed)
            .map(|(id, _)| id.clone())
        {
            sessions.remove(&oldest_id);
            info!("Removed oldest session {} to make room", oldest_id);
        }
    }

    // Create new session
    let new_id = session_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    let memory = MemoryManager::new(&state.config.memory)
        .map_err(|e| AppError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let agent_config = AgentConfig {
        model: state.config.agent.default_model.clone(),
        context_window: state.config.agent.context_window,
        reserve_tokens: state.config.agent.reserve_tokens,
    };

    let mut agent = Agent::new(agent_config, &state.config, memory)
        .await
        .map_err(|e| AppError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    agent
        .new_session()
        .await
        .map_err(|e| AppError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    sessions.insert(
        new_id.clone(),
        SessionEntry {
            agent,
            last_accessed: Instant::now(),
        },
    );

    info!("Created new session: {}", new_id);
    Ok(new_id)
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
    active_sessions: usize,
}

async fn status(State(state): State<Arc<AppState>>) -> Json<StatusResponse> {
    let memory = MemoryManager::new(&state.config.memory).ok();
    let sessions = state.sessions.lock().await;

    Json(StatusResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
        model: state.config.agent.default_model.clone(),
        memory_chunks: memory.and_then(|m| m.chunk_count().ok()).unwrap_or(0),
        active_sessions: sessions.len(),
    })
}

// Session management endpoints
#[derive(Deserialize)]
struct CreateSessionRequest {
    session_id: Option<String>,
}

#[derive(Serialize)]
struct SessionResponse {
    session_id: String,
    model: String,
}

async fn create_session(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CreateSessionRequest>,
) -> Response {
    match get_or_create_session(&state, request.session_id).await {
        Ok(session_id) => Json(SessionResponse {
            session_id,
            model: state.config.agent.default_model.clone(),
        })
        .into_response(),
        Err(e) => e.into_response(),
    }
}

#[derive(Serialize)]
struct SessionInfo {
    session_id: String,
    idle_seconds: u64,
}

#[derive(Serialize)]
struct ListSessionsResponse {
    sessions: Vec<SessionInfo>,
}

async fn list_sessions(State(state): State<Arc<AppState>>) -> Json<ListSessionsResponse> {
    let sessions = state.sessions.lock().await;

    let session_list: Vec<SessionInfo> = sessions
        .iter()
        .map(|(id, entry)| SessionInfo {
            session_id: id.clone(),
            idle_seconds: entry.last_accessed.elapsed().as_secs(),
        })
        .collect();

    Json(ListSessionsResponse {
        sessions: session_list,
    })
}

// Delete a session
async fn delete_session(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
) -> Response {
    let mut sessions = state.sessions.lock().await;

    if sessions.remove(&session_id).is_some() {
        info!("Deleted session: {}", session_id);
        Json(json!({"deleted": true, "session_id": session_id})).into_response()
    } else {
        AppError(StatusCode::NOT_FOUND, "Session not found".to_string()).into_response()
    }
}

// Get session status
#[derive(Serialize)]
struct SessionStatusResponse {
    session_id: String,
    model: String,
    message_count: usize,
    token_count: usize,
    idle_seconds: u64,
    api_input_tokens: u64,
    api_output_tokens: u64,
}

async fn get_session_status(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
) -> Response {
    let sessions = state.sessions.lock().await;

    match sessions.get(&session_id) {
        Some(entry) => {
            let status = entry.agent.session_status();
            Json(SessionStatusResponse {
                session_id,
                model: entry.agent.model().to_string(),
                message_count: status.message_count,
                token_count: status.token_count,
                idle_seconds: entry.last_accessed.elapsed().as_secs(),
                api_input_tokens: status.api_input_tokens,
                api_output_tokens: status.api_output_tokens,
            })
            .into_response()
        }
        None => AppError(StatusCode::NOT_FOUND, "Session not found".to_string()).into_response(),
    }
}

// Chat endpoint
#[derive(Deserialize)]
struct ChatRequest {
    message: String,
    session_id: Option<String>,
    #[allow(dead_code)] // TODO: implement model switching per request
    model: Option<String>,
}

#[derive(Serialize)]
struct ChatResponse {
    response: String,
    session_id: String,
    model: String,
}

async fn chat(State(state): State<Arc<AppState>>, Json(request): Json<ChatRequest>) -> Response {
    // Get or create session
    let session_id = match get_or_create_session(&state, request.session_id).await {
        Ok(id) => id,
        Err(e) => return e.into_response(),
    };

    // Get agent from session
    let mut sessions = state.sessions.lock().await;
    let entry = match sessions.get_mut(&session_id) {
        Some(e) => e,
        None => {
            return AppError(StatusCode::NOT_FOUND, "Session not found".to_string()).into_response()
        }
    };

    entry.last_accessed = Instant::now();

    match entry.agent.chat(&request.message).await {
        Ok(response) => Json(ChatResponse {
            response,
            session_id,
            model: entry.agent.model().to_string(),
        })
        .into_response(),
        Err(e) => AppError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

// Streaming chat endpoint (SSE) with tool support
async fn chat_stream(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatRequest>,
) -> Response {
    // Get or create session first (outside the stream)
    let session_id = match get_or_create_session(&state, request.session_id).await {
        Ok(id) => id,
        Err(e) => return e.into_response(),
    };

    let state_clone = state.clone();
    let message = request.message.clone();

    let stream = async_stream::stream! {
        // Send session_id first
        yield Ok::<Event, Infallible>(Event::default().data(json!({"type": "session", "session_id": session_id}).to_string()));

        let mut sessions = state_clone.sessions.lock().await;
        let entry = match sessions.get_mut(&session_id) {
            Some(e) => e,
            None => {
                yield Ok(Event::default().data(json!({"error": "Session not found"}).to_string()));
                return;
            }
        };

        entry.last_accessed = Instant::now();

        // Use streaming with tools
        match entry.agent.chat_stream_with_tools(&message).await {
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

    Sse::new(stream).into_response()
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
    /// Start or resume a session
    #[serde(rename = "session")]
    Session { session_id: Option<String> },
    /// Chat message (uses tool loop, returns complete response)
    /// For streaming, use the SSE endpoint at /api/chat/stream
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
    ToolEnd {
        name: String,
        id: String,
        output: String,
    },
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

    debug!("WebSocket client connected");

    // Track current session for this connection
    let mut current_session_id: Option<String> = None;

    // Process incoming messages
    while let Some(msg) = receiver.next().await {
        match msg {
            Ok(WsMessage::Text(text)) => {
                // Parse incoming message
                match serde_json::from_str::<WsIncoming>(&text) {
                    Ok(WsIncoming::Session { session_id }) => {
                        // Create or resume session
                        match get_or_create_session(&state, session_id).await {
                            Ok(id) => {
                                current_session_id = Some(id.clone());
                                let connected = WsOutgoing::Connected { session_id: id };
                                if let Ok(json) = serde_json::to_string(&connected) {
                                    let _ = sender.send(WsMessage::Text(json.into())).await;
                                }
                            }
                            Err(e) => {
                                let error = WsOutgoing::Error {
                                    message: format!("Failed to create session: {}", e.1),
                                };
                                if let Ok(json) = serde_json::to_string(&error) {
                                    let _ = sender.send(WsMessage::Text(json.into())).await;
                                }
                            }
                        }
                    }
                    Ok(WsIncoming::Chat { message }) => {
                        // Ensure we have a session
                        let session_id = match &current_session_id {
                            Some(id) => id.clone(),
                            None => {
                                // Auto-create session if none exists
                                match get_or_create_session(&state, None).await {
                                    Ok(id) => {
                                        current_session_id = Some(id.clone());
                                        // Notify client of new session
                                        let connected = WsOutgoing::Connected {
                                            session_id: id.clone(),
                                        };
                                        if let Ok(json) = serde_json::to_string(&connected) {
                                            let _ = sender.send(WsMessage::Text(json.into())).await;
                                        }
                                        id
                                    }
                                    Err(e) => {
                                        let error = WsOutgoing::Error {
                                            message: format!("Failed to create session: {}", e.1),
                                        };
                                        if let Ok(json) = serde_json::to_string(&error) {
                                            let _ = sender.send(WsMessage::Text(json.into())).await;
                                        }
                                        continue;
                                    }
                                }
                            }
                        };

                        debug!("WebSocket chat [{}]: {}", session_id, message);

                        // Process chat
                        let mut sessions = state.sessions.lock().await;
                        let entry = match sessions.get_mut(&session_id) {
                            Some(e) => e,
                            None => {
                                let error = WsOutgoing::Error {
                                    message: "Session not found".to_string(),
                                };
                                if let Ok(json) = serde_json::to_string(&error) {
                                    let _ = sender.send(WsMessage::Text(json.into())).await;
                                }
                                current_session_id = None;
                                continue;
                            }
                        };

                        entry.last_accessed = Instant::now();

                        match entry.agent.chat(&message).await {
                            Ok(response) => {
                                // Send response as content
                                let content = WsOutgoing::Content { delta: response };
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
