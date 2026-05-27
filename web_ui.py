import json
import os
import shutil
import shlex
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_ROOT / "static"
GENERATED_CONFIG_DIR = PROJECT_ROOT / "generated_configs"
LOG_DIR = PROJECT_ROOT / "log"
EVIDENCE_ROOT = PROJECT_ROOT / "zk_evidence"
ZK_WEIGHTS_ROOT = PROJECT_ROOT / "zk_weights"

DEFAULT_PORT = 8001
DEFAULT_CONDA_ENV = os.environ.get("WEB_UI_CONDA_ENV", os.environ.get("CONDA_DEFAULT_ENV", "base"))
CONDA_SH = Path(os.environ.get("CONDA_SH", str(Path.home() / "miniconda3/etc/profile.d/conda.sh")))
SERVER_LOG = LOG_DIR / "web_ui_gac_server.log"
WEB_UI_BASE_URL = os.environ.get("WEB_UI_BASE_URL", "http://127.0.0.1:7860")

app = FastAPI(title="GaC zkhook Local Control UI")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

server_process: Optional[subprocess.Popen] = None
server_info: Dict[str, Any] = {
    "port": DEFAULT_PORT,
    "config_path": "",
    "started_at": None,
    "pid": None,
    "selected_models": [],
}

generation_history: Dict[str, Dict[str, Any]] = {}


MODEL_PRESETS = [
    {
        "name": "qwen-new-7B",
        "weight": str(PROJECT_ROOT / "weights" / "qwen-new-7B"),
        "max_memory": "22GiB",
        "num_gpus": 1,
        "score": 100,
        "priority": "supportive",
        "quantization": "none",
    },
    {
        "name": "Qwen1.5-7B-Chat",
        "weight": str(PROJECT_ROOT / "weights" / "Qwen1.5-7B-Chat"),
        "max_memory": "22GiB",
        "num_gpus": 1,
        "score": 100,
        "priority": "supportive",
        "quantization": "none",
    },
    {
        "name": "Qwen1.5-1.8B-Chat",
        "weight": str(PROJECT_ROOT / "weights" / "Qwen1.5-1.8B-Chat"),
        "max_memory": "22GiB",
        "num_gpus": 1,
        "score": 100,
        "priority": "supportive",
        "quantization": "none",
    },
    {
        "name": "DeepSeek-R1-Distill-Qwen-7B",
        "weight": str(PROJECT_ROOT / "weights" / "DeepSeek-R1-Distill-Qwen-7B"),
        "max_memory": "22GiB",
        "num_gpus": 1,
        "score": 100,
        "priority": "supportive",
        "quantization": "none",
    },
]


class ModelConfig(BaseModel):
    name: str
    weight: str
    max_memory: str = "22GiB"
    num_gpus: float = 1
    score: float = 100
    priority: str = "supportive"
    quantization: str = "none"


class ConfigRequest(BaseModel):
    models: List[ModelConfig]
    norm_type: str = "average"
    threshold: float = 1.0
    filename: Optional[str] = None


class StartServerRequest(BaseModel):
    config_path: str
    port: int = DEFAULT_PORT
    host: str = "127.0.0.1"
    use_model_name_evidence: bool = True


class GenerateRequest(BaseModel):
    prompt: str
    request_id: Optional[str] = None
    max_new_tokens: int = 1
    min_new_tokens: int = 1
    apply_chat_template: bool = False
    until: Optional[List[str]] = None
    port: int = DEFAULT_PORT


class ProveRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    request_id: str
    model_name: str
    weights_dir: Optional[str] = None


class VerifyRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    proof_file: str
    model_name: Optional[str] = None
    tables: Optional[str] = None


class CleanupEvidenceRequest(BaseModel):
    request_id: str


REQUIRED_WEIGHT_FILES = [
    "self_attn.q_proj.weight.bin",
    "self_attn.k_proj.weight.bin",
    "self_attn.v_proj.weight.bin",
    "self_attn.o_proj.weight.bin",
    "mlp.gate_proj.weight.bin",
    "mlp.up_proj.weight.bin",
    "mlp.down_proj.weight.bin",
]


def _project_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _shell_join(parts: List[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _curl_command(method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> str:
    cmd = ["curl", "-sS"]
    if method != "GET":
        cmd.extend(["-X", method, "-H", "Content-Type: application/json"])
        if payload is not None:
            cmd.extend(["-d", json.dumps(payload, ensure_ascii=False)])
    cmd.append(f"{WEB_UI_BASE_URL}{path}")
    return _shell_join(cmd)


def _conda_prefix() -> str:
    if CONDA_SH.exists():
        return f"source {shlex.quote(str(CONDA_SH))} && conda activate {shlex.quote(DEFAULT_CONDA_ENV)}"
    return f"conda activate {shlex.quote(DEFAULT_CONDA_ENV)}"


def _conda_command(parts: List[str]) -> List[str]:
    return ["/bin/bash", "-lc", f"{_conda_prefix()} && {_shell_join(parts)}"]


def _run_command(cmd: List[str], timeout: Optional[int] = None) -> Dict[str, Any]:
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "elapsed_seconds": round(time.time() - started, 3),
        "ok": proc.returncode == 0,
    }


def _run_conda_command(cmd: List[str], timeout: Optional[int] = None) -> Dict[str, Any]:
    return _run_command(_conda_command(cmd), timeout=timeout)


def _service_url(port: int, path: str) -> str:
    return f"http://127.0.0.1:{port}{path}"


def _server_running() -> bool:
    global server_process
    return server_process is not None and server_process.poll() is None


def _check_service_ready(port: int) -> bool:
    try:
        res = requests.get(_service_url(port, "/status"), timeout=1.5)
        return res.ok and res.json().get("status") == "ready"
    except Exception:
        return False


def _read_tail(path: Path, limit: int = 12000) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    return data[-limit:].decode("utf-8", errors="replace")


def _model_size_for_tables(model_name: str) -> str:
    return "1.8B" if "1.8B" in model_name else "7B"


def _weights_dir(model_name: str, override: Optional[str] = None) -> Path:
    return _project_path(override) if override else ZK_WEIGHTS_ROOT / model_name


def _request_evidence_dir(request_id: str) -> Path:
    if not request_id or "/" in request_id or "\\" in request_id or request_id in {".", ".."}:
        raise HTTPException(status_code=400, detail="非法 request_id。")
    request_dir = (EVIDENCE_ROOT / request_id).resolve()
    root = EVIDENCE_ROOT.resolve()
    if root not in request_dir.parents:
        raise HTTPException(status_code=400, detail="非法 evidence 路径。")
    return request_dir


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() and path.is_file() else 0


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def _gpu_compute_apps() -> List[Dict[str, Any]]:
    result = _run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    apps = []
    if not result["ok"]:
        return apps
    for line in result["stdout"].splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                used_memory = int(parts[2])
            except ValueError:
                used_memory = 0
            apps.append(
                {
                    "pid": parts[0],
                    "process_name": parts[1],
                    "used_memory_mb": used_memory,
                }
            )
    return apps


def _human_bytes(num: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{num} B"


def _parse_meta(meta_file: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not meta_file.exists():
        return data
    for line in meta_file.read_text(errors="replace").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip()
    return data


def _check_item(name: str, ok: bool, message: str, severity: str = "error") -> Dict[str, Any]:
    return {"name": name, "ok": ok, "message": message, "severity": severity}


def _preflight_prove(req: ProveRequest) -> Dict[str, Any]:
    checks = []
    evidence = _scan_evidence(req.request_id)
    model_evidence = next((m for m in evidence["models"] if m["model_name"] == req.model_name), None)
    weights_dir = _weights_dir(req.model_name, req.weights_dir)
    tables = PROJECT_ROOT / "zkhook" / f"tables_{_model_size_for_tables(req.model_name)}.bin"
    transformer_prove = PROJECT_ROOT / "zkhook" / "transformer_prove"
    transformer_verify = PROJECT_ROOT / "zkhook" / "transformer_verify"
    gpu_apps = _gpu_compute_apps()
    service_running = _server_running()
    service_ready = _check_service_ready(int(server_info.get("port") or DEFAULT_PORT))

    checks.append(_check_item("evidence_dir", evidence["exists"], f"evidence 目录: {evidence['path']}"))
    checks.append(_check_item("model_evidence", model_evidence is not None, f"模型 evidence: {req.model_name}"))
    if model_evidence:
        for key in ("input_int", "rms_inv", "attention", "ffn", "attn_meta", "ffn_meta"):
            checks.append(
                _check_item(
                    f"evidence_{key}",
                    bool(model_evidence["checks"].get(key)),
                    f"{req.model_name} evidence 检查: {key}",
                )
            )
        checks.append(
            _check_item(
                "no_weight_bin_in_evidence",
                not model_evidence["has_weight_bins"],
                "evidence 中不应包含 weight_*.bin",
            )
        )

    checks.append(_check_item("zk_weights_dir", weights_dir.exists(), f"zk_weights 目录: {weights_dir}"))
    if weights_dir.exists():
        for filename in REQUIRED_WEIGHT_FILES:
            checks.append(
                _check_item(
                    f"weight_{filename}",
                    (weights_dir / filename).exists(),
                    f"长期权重文件: {filename}",
                )
            )

    checks.append(_check_item("transformer_prove", transformer_prove.exists(), f"CUDA prove: {transformer_prove}"))
    checks.append(_check_item("transformer_verify", transformer_verify.exists(), f"CUDA verify: {transformer_verify}", "warning"))
    checks.append(_check_item("tables", tables.exists(), f"lookup table: {tables}"))
    checks.append(
        _check_item(
            "fusion_service_stopped",
            not service_running and not service_ready,
            "建议先停止融合服务再生成 proof，避免 CUDA/Ray 并发冲突。",
            "warning",
        )
    )
    checks.append(
        _check_item(
            "gpu_free",
            len(gpu_apps) == 0,
            f"GPU 计算进程: {len(gpu_apps)} 个。proof 前建议为空。",
            "warning",
        )
    )

    errors = [c for c in checks if not c["ok"] and c["severity"] == "error"]
    warnings = [c for c in checks if not c["ok"] and c["severity"] == "warning"]
    return {
        "ok": not errors,
        "safe_to_run": not errors and not warnings,
        "request_id": req.request_id,
        "model_name": req.model_name,
        "weights_dir": str(weights_dir),
        "tables": str(tables),
        "proof_file": str(EVIDENCE_ROOT / req.request_id / f"transformer_proof_{req.model_name}.bin"),
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "gpu_apps": gpu_apps,
        "service": {
            "running": service_running,
            "ready": service_ready,
            "pid": server_process.pid if service_running and server_process else server_info.get("pid"),
        },
        "command": _curl_command("POST", "/ui/prove/preflight", req.model_dump()),
    }


def _selected_models_from_config(config_path: Path) -> List[Dict[str, Any]]:
    if not config_path.exists():
        return []
    try:
        config = yaml.safe_load(config_path.read_text()) or {}
    except Exception:
        return []
    models = []
    for item in config.get("CONFIG_API_SERVER", []) or []:
        models.append(
            {
                "name": item.get("name", ""),
                "weight": item.get("weight", ""),
                "score": item.get("score"),
                "priority": item.get("priority"),
                "quantization": item.get("quantization", "none"),
                "num_gpus": item.get("num_gpus"),
                "max_memory": item.get("max_memory", {}),
            }
        )
    return models


def _scan_evidence(request_id: str) -> Dict[str, Any]:
    request_dir = _request_evidence_dir(request_id)
    models = []
    total_files = 0
    total_weight_bins = 0
    if request_dir.exists():
        for model_dir in sorted([p for p in request_dir.iterdir() if p.is_dir()]):
            layer_dir = model_dir / "layer_0"
            attn_dir = layer_dir / "attention"
            ffn_dir = layer_dir / "ffn"
            rms_dir = layer_dir / "rmsnorm"
            attn_meta_files = sorted(attn_dir.glob("attn_meta_*.txt"))
            ffn_meta_files = sorted(ffn_dir.glob("ffn_meta_*.txt"))
            rms_meta_files = sorted(rms_dir.glob("rmsnorm_meta_*.txt"))
            meta = {}
            for files in (attn_meta_files[-1:], ffn_meta_files[-1:], rms_meta_files[-1:]):
                if files:
                    meta.update(_parse_meta(files[0]))
            weight_bins = list(layer_dir.rglob("weight_*.bin")) if layer_dir.exists() else []
            files = [p for p in layer_dir.rglob("*") if p.is_file()] if layer_dir.exists() else []
            proof_file = request_dir / f"transformer_proof_{model_dir.name}.bin"
            model_bytes = _dir_size(layer_dir)
            total_files += len(files)
            total_weight_bins += len(weight_bins)
            models.append(
                {
                    "model_name": model_dir.name,
                    "path": str(layer_dir),
                    "file_count": len(files),
                    "bytes": model_bytes,
                    "human_size": _human_bytes(model_bytes),
                    "weight_bin_count": len(weight_bins),
                    "has_weight_bins": len(weight_bins) > 0,
                    "checks": {
                        "input_int": (layer_dir / "input_int.bin").exists(),
                        "rms_inv": (layer_dir / "rms_inv.bin").exists(),
                        "rmsnorm": rms_dir.exists(),
                        "attention": attn_dir.exists(),
                        "ffn": ffn_dir.exists(),
                        "attn_meta": bool(attn_meta_files),
                        "ffn_meta": bool(ffn_meta_files),
                    },
                    "meta": meta,
                    "proof": {
                        "path": str(proof_file),
                        "exists": proof_file.exists(),
                        "bytes": _file_size(proof_file),
                        "human_size": _human_bytes(_file_size(proof_file)),
                    },
                }
            )
    proof_files = sorted(request_dir.glob("transformer_proof_*.bin")) if request_dir.exists() else []
    return {
        "request_id": request_id,
        "path": str(request_dir),
        "command": _curl_command("GET", f"/ui/evidence/{request_id}"),
        "exists": request_dir.exists(),
        "file_count": total_files + len(proof_files),
        "total_bytes": _dir_size(request_dir),
        "human_total_size": _human_bytes(_dir_size(request_dir)),
        "weight_bin_count": total_weight_bins,
        "has_weight_bins": total_weight_bins > 0,
        "proof_bytes": sum(_file_size(p) for p in proof_files),
        "proof_files": [
            {"path": str(p), "bytes": _file_size(p), "human_size": _human_bytes(_file_size(p))}
            for p in proof_files
        ],
        "models": models,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/ui/status")
def ui_status() -> Dict[str, Any]:
    ready = _check_service_ready(int(server_info.get("port") or DEFAULT_PORT))
    running = _server_running()
    return {
        "web_ui": "ready",
        "project_root": str(PROJECT_ROOT),
        "service": {
            "running": running,
            "ready": ready,
            "pid": server_process.pid if running and server_process else server_info.get("pid"),
            "port": server_info.get("port"),
            "config_path": server_info.get("config_path"),
            "started_at": server_info.get("started_at"),
            "selected_models": server_info.get("selected_models", []),
            "log_tail": _read_tail(SERVER_LOG, 8000),
        },
    }


@app.get("/ui/gpu")
def ui_gpu() -> Dict[str, Any]:
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    apps = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    gpu_result = _run_command(query)
    apps_result = _run_command(apps)
    gpus = []
    apps_list = []
    if gpu_result["ok"]:
        for line in gpu_result["stdout"].splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append(
                    {
                        "index": parts[0],
                        "name": parts[1],
                        "memory_total_mb": int(parts[2]),
                        "memory_used_mb": int(parts[3]),
                        "utilization": int(parts[4]),
                    }
                )
    if apps_result["ok"]:
        for line in apps_result["stdout"].splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                apps_list.append(
                    {
                        "pid": parts[0],
                        "process_name": parts[1],
                        "used_memory_mb": int(parts[2]),
                    }
                )
    return {
        "gpus": gpus,
        "apps": apps_list,
        "apps_raw": apps_result["stdout"],
        "errors": gpu_result["stderr"] + apps_result["stderr"],
    }


@app.get("/ui/models")
def ui_models() -> Dict[str, Any]:
    configs = sorted([str(p.relative_to(PROJECT_ROOT)) for p in PROJECT_ROOT.glob("*.yaml")])
    generated = sorted(str(p.relative_to(PROJECT_ROOT)) for p in GENERATED_CONFIG_DIR.glob("*.yaml"))
    return {"models": MODEL_PRESETS, "configs": configs, "generated_configs": generated}


@app.post("/ui/config")
def ui_config(req: ConfigRequest) -> Dict[str, Any]:
    if len(req.models) < 2:
        raise HTTPException(status_code=400, detail="至少需要选择两个模型。")
    primary_count = sum(1 for m in req.models if m.priority == "primary")
    if primary_count > 1:
        raise HTTPException(status_code=400, detail="最多只能有一个 primary 模型。")
    for m in req.models:
        if m.priority not in {"supportive", "primary"}:
            raise HTTPException(status_code=400, detail=f"非法 priority: {m.priority}")
        if m.quantization not in {"none", "8bit", "4bit"}:
            raise HTTPException(status_code=400, detail=f"非法 quantization: {m.quantization}")

    data = {
        "NORM_TYPE_API_SERVER": req.norm_type,
        "THRESHOLD_API_SERVER": req.threshold,
        "CONFIG_API_SERVER": [
            {
                "weight": m.weight,
                "max_memory": {0: m.max_memory},
                "num_gpus": m.num_gpus,
                "name": m.name,
                "score": m.score,
                "priority": m.priority,
                "quantization": m.quantization,
            }
            for m in req.models
        ],
    }
    GENERATED_CONFIG_DIR.mkdir(exist_ok=True)
    filename = req.filename or f"ui_config_{int(time.time())}.yaml"
    if not filename.endswith(".yaml"):
        filename += ".yaml"
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
    path = GENERATED_CONFIG_DIR / safe_name
    yaml_text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    path.write_text(yaml_text)
    payload = req.model_dump()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "yaml": yaml_text,
        "command": _curl_command("POST", "/ui/config", payload),
    }


@app.post("/ui/start-server")
def ui_start_server(req: StartServerRequest) -> Dict[str, Any]:
    global server_process, server_info
    if _server_running():
        raise HTTPException(status_code=409, detail="融合 API 服务已经在运行。")

    config_path = _project_path(req.config_path)
    if not config_path.exists():
        raise HTTPException(status_code=404, detail=f"配置文件不存在: {config_path}")

    selected_models = _selected_models_from_config(config_path)
    LOG_DIR.mkdir(exist_ok=True)
    log_file = SERVER_LOG.open("ab")
    env = os.environ.copy()
    env["ZK_EVIDENCE_USE_MODEL_NAME"] = "1" if req.use_model_name_evidence else "0"
    app_cmd = [
        "python",
        str(PROJECT_ROOT / "gac_api_server.py"),
        "--config-path",
        str(config_path),
        "--host",
        req.host,
        "--port",
        str(req.port),
    ]
    cmd = _conda_command(app_cmd)
    server_process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    server_info = {
        "port": req.port,
        "config_path": str(config_path),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pid": server_process.pid,
        "selected_models": selected_models,
        "command": " ".join(cmd),
        "conda_env": DEFAULT_CONDA_ENV,
    }
    return {
        "ok": True,
        "pid": server_process.pid,
        "command": " ".join(cmd),
        "env": {"ZK_EVIDENCE_USE_MODEL_NAME": env["ZK_EVIDENCE_USE_MODEL_NAME"]},
        "log": str(SERVER_LOG),
    }


@app.post("/ui/stop-server")
def ui_stop_server() -> Dict[str, Any]:
    global server_process, server_info
    stopped = False
    if _server_running() and server_process:
        try:
            os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
            try:
                server_process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)
            stopped = True
        except ProcessLookupError:
            stopped = False
    elif server_info.get("pid"):
        try:
            os.killpg(os.getpgid(int(server_info["pid"])), signal.SIGTERM)
            stopped = True
        except Exception:
            stopped = False
    server_process = None
    server_info.update({"pid": None, "started_at": None})
    ray_stop = _run_conda_command(["ray", "stop", "--force"], timeout=30)
    return {
        "ok": True,
        "stopped_process": stopped,
        "command": "kill service process group; ray stop --force",
        "ray_stop": ray_stop,
    }


@app.post("/ui/generate")
def ui_generate(req: GenerateRequest) -> Dict[str, Any]:
    request_id = req.request_id or f"ui_{uuid.uuid4().hex[:10]}"
    payload = {
        "messages_list": [[{"role": "user", "content": req.prompt}]],
        "max_new_tokens": req.max_new_tokens,
        "min_new_tokens": max(0, min(req.min_new_tokens, req.max_new_tokens)),
        "apply_chat_template": req.apply_chat_template,
        "request_id": request_id,
    }
    if req.until:
        payload["until"] = req.until
    command = _curl_command(
        "POST",
        "/ui/generate",
        {
            "prompt": req.prompt,
            "request_id": request_id,
            "max_new_tokens": req.max_new_tokens,
            "min_new_tokens": req.min_new_tokens,
            "apply_chat_template": req.apply_chat_template,
            "until": req.until,
            "port": req.port,
        },
    )

    started = time.time()
    try:
        res = requests.post(_service_url(req.port, "/api/generate/"), json=payload, timeout=900)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"生成请求失败: {exc}") from exc
    elapsed = time.time() - started
    if not res.ok:
        raise HTTPException(status_code=res.status_code, detail=res.text)
    data = res.json()
    answer = (data.get("response") or [""])[0]
    generated_tokens = max(req.max_new_tokens, 1)
    generation_history[request_id] = {
        "prompt": req.prompt,
        "answer": answer,
        "max_new_tokens": req.max_new_tokens,
        "min_new_tokens": req.min_new_tokens,
        "generated_tokens": generated_tokens,
        "elapsed_seconds": elapsed,
        "log_tail": _read_tail(SERVER_LOG, 12000),
    }
    return {
        "ok": True,
        "request_id": request_id,
        "answer": answer,
        "generated_tokens": generated_tokens,
        "elapsed_seconds": round(elapsed, 3),
        "command": command,
        "upstream_url": _service_url(req.port, "/api/generate/"),
        "upstream_payload": payload,
        "log_tail": generation_history[request_id]["log_tail"],
    }


@app.get("/ui/evidence/{request_id}")
def ui_evidence(request_id: str) -> Dict[str, Any]:
    return _scan_evidence(request_id)


@app.post("/ui/prove/preflight")
def ui_prove_preflight(req: ProveRequest) -> Dict[str, Any]:
    return _preflight_prove(req)


@app.post("/ui/prove")
def ui_prove(req: ProveRequest) -> Dict[str, Any]:
    preflight = _preflight_prove(req)
    if not preflight["ok"]:
        raise HTTPException(status_code=400, detail={"message": "proof 前置检查失败。", "preflight": preflight})
    weights_dir = _weights_dir(req.model_name, req.weights_dir)
    if not weights_dir.exists():
        raise HTTPException(status_code=404, detail=f"zk_weights 不存在: {weights_dir}")
    cmd = [
        "python",
        str(PROJECT_ROOT / "verify.py"),
        "--prove",
        "--verify",
        "--request-id",
        req.request_id,
        "--model-size",
        req.model_name,
        "--weights-dir",
        str(weights_dir),
    ]
    result = _run_conda_command(cmd, timeout=1200)
    proof_file = EVIDENCE_ROOT / req.request_id / f"transformer_proof_{req.model_name}.bin"
    result["proof_file"] = str(proof_file)
    result["proof_bytes"] = _file_size(proof_file)
    result["proof_human_size"] = _human_bytes(result["proof_bytes"])
    result["ui_command"] = _curl_command("POST", "/ui/prove", req.model_dump())
    result["preflight"] = preflight
    return result


@app.post("/ui/verify")
def ui_verify(req: VerifyRequest) -> Dict[str, Any]:
    proof_file = _project_path(req.proof_file)
    if not proof_file.exists():
        raise HTTPException(status_code=404, detail=f"proof 文件不存在: {proof_file}")
    if req.tables:
        tables = _project_path(req.tables)
    else:
        model_name = req.model_name or proof_file.stem.replace("transformer_proof_", "")
        tables = PROJECT_ROOT / "zkhook" / f"tables_{_model_size_for_tables(model_name)}.bin"
    if not tables.exists():
        raise HTTPException(status_code=404, detail=f"lookup table 文件不存在: {tables}")
    cmd = [
        "python",
        str(PROJECT_ROOT / "verify.py"),
        "--verify",
        "--proof-file",
        str(proof_file),
        "--tables",
        str(tables),
    ]
    result = _run_conda_command(cmd, timeout=600)
    result["proof_file"] = str(proof_file)
    result["tables"] = str(tables)
    result["verified"] = result["ok"] and ("验证通过" in result["stdout"] or "Verification passed" in result["stdout"])
    result["ui_command"] = _curl_command("POST", "/ui/verify", req.model_dump())
    return result


@app.get("/ui/metrics/{request_id}")
def ui_metrics(request_id: str) -> Dict[str, Any]:
    evidence = _scan_evidence(request_id)
    history = generation_history.get(request_id, {})
    generated_tokens = int(history.get("generated_tokens") or 1)
    total_bytes = int(evidence["total_bytes"])
    proof_bytes = int(evidence["proof_bytes"])
    evidence_bytes = max(total_bytes - proof_bytes, 0)
    return {
        "request_id": request_id,
        "command": _curl_command("GET", f"/ui/metrics/{request_id}"),
        "generated_tokens": generated_tokens,
        "evidence_bytes": evidence_bytes,
        "evidence_human_size": _human_bytes(evidence_bytes),
        "proof_bytes": proof_bytes,
        "proof_human_size": _human_bytes(proof_bytes),
        "total_bytes": total_bytes,
        "total_human_size": _human_bytes(total_bytes),
        "evidence_bytes_per_token": evidence_bytes / max(generated_tokens, 1),
        "proof_bytes_per_token": proof_bytes / max(generated_tokens, 1),
        "total_bytes_per_token": total_bytes / max(generated_tokens, 1),
        "evidence_per_token_human": _human_bytes(int(evidence_bytes / max(generated_tokens, 1))),
        "proof_per_token_human": _human_bytes(int(proof_bytes / max(generated_tokens, 1))),
        "total_per_token_human": _human_bytes(int(total_bytes / max(generated_tokens, 1))),
        "models": evidence["models"],
        "proof_files": evidence["proof_files"],
    }


@app.post("/ui/cleanup-evidence")
def ui_cleanup_evidence(req: CleanupEvidenceRequest) -> Dict[str, Any]:
    request_dir = _request_evidence_dir(req.request_id)
    zk_weights = ZK_WEIGHTS_ROOT.resolve()
    if request_dir == zk_weights or zk_weights in request_dir.parents:
        raise HTTPException(status_code=400, detail="拒绝删除 zk_weights。")
    existed = request_dir.exists()
    bytes_before = _dir_size(request_dir)
    if existed:
        shutil.rmtree(request_dir)
    return {
        "ok": True,
        "request_id": req.request_id,
        "deleted": existed,
        "path": str(request_dir),
        "bytes_deleted": bytes_before,
        "human_deleted": _human_bytes(bytes_before),
        "zk_weights_preserved": str(ZK_WEIGHTS_ROOT),
        "command": _curl_command("POST", "/ui/cleanup-evidence", req.model_dump()),
    }


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the local zkhook Web UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind.")
    args = parser.parse_args()
    WEB_UI_BASE_URL = os.environ.get("WEB_UI_BASE_URL", f"http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
