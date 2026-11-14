use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::mpsc;

use anyhow::{Context, Result};
use axum::extract::State;
use axum::http::{Method, StatusCode};
use axum::routing::{get, post};
use axum::{Json, Router};
use candle_core::Device;
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::oneshot;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use wisp_runtime::inference::{DecideRequest, DecideResult, WispEngine};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DeviceArg {
    Cpu,
    Cuda,
}

#[derive(Debug, Parser)]
#[command(about = "Native Rust/Candle inference API for Wisp 4B v0.1.2")]
struct Args {
    #[arg(long, env = "WISP_MODEL_DIR")]
    model_dir: PathBuf,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8782)]
    port: u16,
    #[arg(long, value_enum, default_value_t = DeviceArg::Cuda)]
    device: DeviceArg,
}

#[derive(Clone)]
struct AppState {
    worker: InferenceWorker,
    device: &'static str,
}

#[derive(Clone)]
struct InferenceWorker {
    sender: mpsc::Sender<InferenceJob>,
}

struct InferenceJob {
    request: DecideRequest,
    reply: oneshot::Sender<Result<DecideResult, String>>,
}

impl InferenceWorker {
    fn start(model_dir: PathBuf, device: DeviceArg) -> Result<Self> {
        let (sender, receiver) = mpsc::channel::<InferenceJob>();
        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<(), String>>(1);
        std::thread::Builder::new()
            .name("wisp-candle-inference".into())
            .spawn(move || {
                // CUDA contexts are thread-local. Creating the device and model in this
                // dedicated thread makes every request use the same valid context.
                let candle_device = match device {
                    DeviceArg::Cpu => Device::Cpu,
                    DeviceArg::Cuda => match Device::new_cuda(0) {
                        Ok(device) => device,
                        Err(error) => {
                            let _ = ready_tx.send(Err(format!("initialize CUDA device 0: {error}")));
                            return;
                        }
                    },
                };
                let mut engine = match WispEngine::load(&model_dir, candle_device) {
                    Ok(engine) => engine,
                    Err(error) => {
                        let _ = ready_tx.send(Err(format!("load Wisp model: {error}")));
                        return;
                    }
                };
                if ready_tx.send(Ok(())).is_err() {
                    return;
                }
                while let Ok(job) = receiver.recv() {
                    let result = engine.decide(job.request).map_err(|error| error.to_string());
                    let _ = job.reply.send(result);
                }
            })
            .context("start dedicated Wisp inference thread")?;
        ready_rx
            .recv()
            .context("wait for Wisp inference worker startup")?
            .map_err(anyhow::Error::msg)?;
        Ok(Self { sender })
    }

    async fn decide(&self, request: DecideRequest) -> Result<DecideResult> {
        let (reply, response) = oneshot::channel();
        self.sender
            .send(InferenceJob { request, reply })
            .map_err(|_| anyhow::anyhow!("Wisp inference worker stopped"))?;
        response
            .await
            .map_err(|_| anyhow::anyhow!("Wisp inference worker dropped its response"))?
            .map_err(anyhow::Error::msg)
    }
}

#[derive(Serialize)]
struct ResultEnvelope {
    result: DecideResult,
}

/// Deliberately small discovery contract for browser clients such as Wisp
/// Benchmark. This only describes this runtime; it does not scan the user's
/// machine or claim to manage unrelated model servers.
#[derive(Serialize)]
struct ModelsEnvelope {
    runtime: &'static str,
    status: &'static str,
    models: Vec<LocalModel>,
}

#[derive(Serialize)]
struct LocalModel {
    id: &'static str,
    label: &'static str,
    base_model: &'static str,
    revision: &'static str,
    tasks: Vec<&'static str>,
    calibration: &'static str,
    automatic_punishment: bool,
    device: &'static str,
}

#[derive(Debug, Deserialize)]
struct TextRequest {
    text: String,
}

#[derive(Serialize)]
struct ModerationEnvelope {
    review: DecideResult,
    category: DecideResult,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let state = AppState {
        worker: InferenceWorker::start(args.model_dir, args.device)?,
        device: match args.device {
            DeviceArg::Cpu => "cpu",
            DeviceArg::Cuda => "cuda:0",
        },
    };
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST])
        .allow_headers(Any)
        // Lets an https page (such as the benchmark site) call this local server in Chrome-based browsers.
        .allow_private_network(true);
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/decide", post(decide))
        .route("/v1/moderate", post(moderate))
        .route("/v1/username", post(username))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state);
    let address: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    let listener = tokio::net::TcpListener::bind(address).await?;
    println!("{}", json!({"status":"listening","address":address.to_string(),"runtime":"candle-rust"}));
    axum::serve(listener, app).with_graceful_shutdown(shutdown()).await?;
    Ok(())
}

async fn health(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "model": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "runtime": "candle-rust",
        "device": state.device,
        "calibration": "uncalibrated",
        "automatic_punishment": false
    }))
}

async fn models(State(state): State<AppState>) -> Json<ModelsEnvelope> {
    Json(ModelsEnvelope {
        runtime: "candle-rust",
        status: "ok",
        models: vec![LocalModel {
            id: "wisp-v0.1.2-4b",
            label: "Wisp v0.1.2 4B",
            base_model: "Qwen/Qwen3.5-4B",
            revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            tasks: vec!["moderation", "username", "report_validity"],
            calibration: "uncalibrated",
            automatic_punishment: false,
            device: state.device,
        }],
    })
}

async fn decide(
    State(state): State<AppState>,
    Json(request): Json<DecideRequest>,
) -> Result<Json<ResultEnvelope>, (StatusCode, Json<Value>)> {
    let result = state.worker.decide(request).await.map_err(bad_request)?;
    Ok(Json(ResultEnvelope { result }))
}

async fn moderate(
    State(state): State<AppState>,
    Json(request): Json<TextRequest>,
) -> Result<Json<ModerationEnvelope>, (StatusCode, Json<Value>)> {
    classify_moderation(state, request.text, "moderation").await
}

async fn username(
    State(state): State<AppState>,
    Json(request): Json<TextRequest>,
) -> Result<Json<ModerationEnvelope>, (StatusCode, Json<Value>)> {
    classify_moderation(state, request.text, "username").await
}

async fn classify_moderation(
    state: AppState,
    text: String,
    task: &'static str,
) -> Result<Json<ModerationEnvelope>, (StatusCode, Json<Value>)> {
    let review = state
        .worker
        .decide(DecideRequest {
            task: Some(task.into()),
            text: text.clone(),
            question: "Does this content require moderator review?".into(),
            options: vec!["REVIEW".into(), "NO_ACTION".into(), "NEED_CONTEXT".into()],
        })
        .await
        .map_err(bad_request)?;
    let category = state
        .worker
        .decide(DecideRequest {
            task: Some(task.into()),
            text,
            question: "What is the primary category?".into(),
            options: ["INSULT", "THREAT", "HARASSMENT", "HATE", "SEXUAL", "DOXXING", "OTHER", "NONE"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
        })
        .await
        .map_err(bad_request)?;
    Ok(Json(ModerationEnvelope { review, category }))
}

fn bad_request(error: anyhow::Error) -> (StatusCode, Json<Value>) {
    (StatusCode::BAD_REQUEST, Json(json!({"error": error.to_string()})))
}

async fn shutdown() {
    let _ = tokio::signal::ctrl_c().await;
}

