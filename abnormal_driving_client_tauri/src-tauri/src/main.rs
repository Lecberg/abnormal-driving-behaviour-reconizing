use serde_json::Value;
use std::{
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::Mutex,
    thread,
};
use tauri::{AppHandle, Emitter, Manager, State};

struct BackendProcess {
    child: Child,
    stdin: ChildStdin,
}

#[derive(Default)]
struct BackendState {
    process: Mutex<Option<BackendProcess>>,
}

fn find_workspace_root(start: &Path) -> Option<PathBuf> {
    for candidate in start.ancestors() {
        if candidate.join("abnormal_driving_client").join("backend_service.py").exists() {
            return Some(candidate.to_path_buf());
        }
    }
    None
}

fn python_works(path: &str) -> bool {
    Command::new(path)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn find_python(workspace_root: &Path) -> String {
    if let Ok(python) = std::env::var("PYTHON") {
        if python_works(&python) {
            return python;
        }
    }

    let mut candidates = vec![
        workspace_root.join(".venv312").join("Scripts").join("python.exe"),
        workspace_root.join(".venv").join("Scripts").join("python.exe"),
    ];

    if let Ok(user_profile) = std::env::var("USERPROFILE") {
        candidates.push(
            PathBuf::from(user_profile)
                .join(".cache")
                .join("codex-runtimes")
                .join("codex-primary-runtime")
                .join("dependencies")
                .join("python")
                .join("python.exe"),
        );
    }

    for candidate in candidates {
        let candidate_text = candidate.to_string_lossy().to_string();
        if candidate.exists() && python_works(&candidate_text) {
            return candidate_text;
        }
    }

    "python".to_string()
}

fn emit_backend_line(app: &AppHandle, line: &str) {
    match serde_json::from_str::<Value>(line) {
        Ok(value) => {
            let _ = app.emit("backend-event", value);
        }
        Err(_) => {
            let _ = app.emit("backend-event", serde_json::json!({
                "event": "error",
                "payload": { "message": line }
            }));
        }
    }
}

#[tauri::command]
fn start_backend(app: AppHandle, state: State<BackendState>) -> Result<(), String> {
    let mut process_guard = state
        .process
        .lock()
        .map_err(|_| "Backend process lock failed".to_string())?;

    if process_guard.is_some() {
        return Ok(());
    }

    let current_dir = std::env::current_dir().map_err(|error| error.to_string())?;
    let workspace_root = find_workspace_root(&current_dir)
        .ok_or_else(|| "Could not find abnormal_driving_client/backend_service.py".to_string())?;
    let backend_script = workspace_root
        .join("abnormal_driving_client")
        .join("backend_service.py");
    let python = find_python(&workspace_root);

    let mut child = Command::new(python)
        .arg("-u")
        .arg(&backend_script)
        .current_dir(&workspace_root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("Failed to start Python backend: {error}"))?;

    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| "Python backend stdin is unavailable".to_string())?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Python backend stdout is unavailable".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "Python backend stderr is unavailable".to_string())?;

    let stdout_app = app.clone();
    thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines().flatten() {
            emit_backend_line(&stdout_app, &line);
        }
    });

    let stderr_app = app.clone();
    thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines().flatten() {
            let _ = stderr_app.emit("backend-event", serde_json::json!({
                "event": "error",
                "payload": { "message": line }
            }));
        }
    });

    *process_guard = Some(BackendProcess { child, stdin });
    Ok(())
}

#[tauri::command]
fn send_backend_command(state: State<BackendState>, command: Value) -> Result<(), String> {
    let mut process_guard = state
        .process
        .lock()
        .map_err(|_| "Backend process lock failed".to_string())?;
    let process = process_guard
        .as_mut()
        .ok_or_else(|| "Backend is not running".to_string())?;
    let line = serde_json::to_string(&command).map_err(|error| error.to_string())?;
    process
        .stdin
        .write_all(line.as_bytes())
        .and_then(|_| process.stdin.write_all(b"\n"))
        .and_then(|_| process.stdin.flush())
        .map_err(|error| format!("Failed to send command to Python backend: {error}"))
}

#[tauri::command]
fn stop_backend(state: State<BackendState>) -> Result<(), String> {
    let mut process_guard = state
        .process
        .lock()
        .map_err(|_| "Backend process lock failed".to_string())?;
    if let Some(mut process) = process_guard.take() {
        let _ = process.stdin.write_all(br#"{"command":"shutdown","payload":{}}"#);
        let _ = process.stdin.write_all(b"\n");
        let _ = process.stdin.flush();
        let _ = process.child.kill();
    }
    Ok(())
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(BackendState::default())
        .setup(|app| {
            let handle = app.handle().clone();
            let state = app.state::<BackendState>();
            start_backend(handle, state).map_err(Box::<dyn std::error::Error>::from)?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            start_backend,
            send_backend_command,
            stop_backend
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                let state = window.state::<BackendState>();
                let _ = stop_backend(state);
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
