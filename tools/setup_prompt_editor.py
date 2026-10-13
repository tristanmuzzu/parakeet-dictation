"""Install the pinned Linux x64 local editor assets; never sends transcripts.

Run once with system Python. Runtime/model files are gitignored. Downloads are
SHA-256 checked before extraction/use. No system packages or services are changed.
"""
import hashlib
import json
import argparse
from pathlib import Path
import platform
import shutil
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1] / ".local-editor"
RUNTIME_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b11146/llama-b11146-bin-ubuntu-x64.tar.gz"
RUNTIME_SHA = "c150306eb16b5ab696f76a8bdf810c35fd98a24e82158742e6fa28f420ff8410"
MODEL_REPO = "unsloth/Qwen3.5-2B-GGUF"
MODEL_FILE = "Qwen3.5-2B-Q4_K_M.gguf"
MODEL_SHA = "aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223"


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def download(url, path, expected):
    if path.exists() and digest(path) == expected:
        return
    temporary = path.with_suffix(path.suffix + ".part")
    print("Downloading", path.name, flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as source, temporary.open("wb") as target:
            size = int(source.headers.get("Content-Length", 0))
            if shutil.disk_usage(path.parent).free < size + 256 * 1024**2:
                raise RuntimeError("Insufficient storage; choose a different --assets-dir")
            while block := source.read(1024 * 1024):
                target.write(block)
        if digest(temporary) != expected:
            raise RuntimeError("Download checksum mismatch: " + path.name)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", "--model-dir", dest="assets_dir", type=Path, default=ROOT)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise SystemExit("This installer is for Linux x64 only.")
    ROOT.mkdir(exist_ok=True)
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    assets = args.assets_dir.resolve()
    archive = assets / "llama-b11146.tar.gz"
    download(RUNTIME_URL, archive, RUNTIME_SHA)
    runtime = assets / "runtime"
    runtime.mkdir(exist_ok=True)
    with tarfile.open(archive) as package:
        package.extractall(runtime, filter="data")
    server = next(runtime.rglob("llama-server"))
    model = assets / MODEL_FILE
    download(f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}", model, MODEL_SHA)
    manifest = dict(server=str(server), model=str(model),
                    runtime="llama.cpp b11146", runtime_sha256=RUNTIME_SHA,
                    model_repo=MODEL_REPO, model_sha256=MODEL_SHA)
    (ROOT / "runtime.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Local editor assets verified and installed.", flush=True)


if __name__ == "__main__":
    main()
