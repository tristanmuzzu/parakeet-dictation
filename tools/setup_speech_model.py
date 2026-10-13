"""Install pinned native CPU speech assets without changing the active model.

Linux/Windows x64; choose a data partition with at least 1 GiB free. Downloads
are public, SHA-256 verified, and contain no user audio or transcripts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tarfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1] / '.local-speech'
VERSION = '0.1.0'
RUNTIMES = {
    'Linux': ('linux-x86_64-cpu.tar.gz', '0f74131d631ad2c694cf0ec53490866bb6461147959589a69fb6fc231944065b'),
    'Windows': ('windows-x86_64-cpu.zip', '5e4ea81046012edcd77fd8848de8eefb5a4ba38cc26f52eb544ab184695a75d6'),
}
MODELS = {
    'nemotron-3.5': ('nvidia/nemotron-3.5-asr-streaming-0.6b',
        'ea30d66debe3740a08b573244286791d423d6b3e',
        'nemotron-3.5-asr-streaming-0.6b.q8_0.gguf',
        '3fc991d3badad7277c11030a7519832cddaf2057aafed6d4b25147e953a070b1', 742090464),
    'nemotron-en': ('nvidia/nemotron-speech-streaming-en-0.6b',
        'ebe59e5a817142986528bbbee5dba8db7b38ed50',
        'nemotron-speech-streaming-en-0.6b.q8_0.gguf',
        'd9a01898d2a611c8764e23a1c2f45e70bbd5a425dc4de93692ac951dd603812d', 699872960),
}


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def download(url, path, sha, expected_size=0):
    if path.is_file() and digest(path) == sha:
        return
    temporary = path.with_suffix(path.suffix + '.part')
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as output:
            size = expected_size or int(response.headers.get('Content-Length', 0))
            if shutil.disk_usage(path.parent).free < size + 256 * 1024**2:
                raise RuntimeError('Insufficient space; choose a different --assets-dir')
            count = 0
            while block := response.read(1024 * 1024):
                output.write(block)
                count += len(block)
            if expected_size and count != expected_size:
                raise RuntimeError('Incomplete download: ' + path.name)
        if digest(temporary) != sha:
            raise RuntimeError('Checksum mismatch: ' + path.name)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-dir', type=Path, default=ROOT)
    parser.add_argument('--model', choices=MODELS, default='nemotron-3.5')
    args = parser.parse_args()
    system = platform.system()
    if system not in RUNTIMES or platform.machine().lower() not in ('x86_64','amd64'):
        raise SystemExit('Native speech installer currently supports Linux/Windows x64.')
    assets = args.assets_dir.resolve()
    assets.mkdir(parents=True, exist_ok=True)
    suffix, runtime_sha = RUNTIMES[system]
    archive = assets / f'nemo-speech-{VERSION}-{suffix}'
    print('Verifying native CPU runtime', flush=True)
    download(f'https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v{VERSION}/{archive.name}',
             archive, runtime_sha)
    runtime = assets / 'runtime'
    runtime.mkdir(exist_ok=True)
    if system == 'Linux':
        with tarfile.open(archive) as package:
            package.extractall(runtime, filter='data')
        library = next(runtime.rglob('libnemo_speech_asr_c.so'))
    else:
        with zipfile.ZipFile(archive) as package:
            for info in package.infolist():
                destination = (runtime / info.filename).resolve()
                if not destination.is_relative_to(runtime.resolve()):
                    raise RuntimeError('Invalid runtime archive path')
            package.extractall(runtime)
        library = next(runtime.rglob('nemo_speech_asr_c.dll'))
    repo, revision, filename, sha, size = MODELS[args.model]
    model = assets / filename
    print('Downloading/verifying', filename, flush=True)
    download(f'https://huggingface.co/{repo}/resolve/{revision}/{filename}',model,sha,size)
    manifest = dict(runtime=f'NeMo-Speech.cpp {VERSION}', runtime_sha256=runtime_sha,
                    library=str(library), model=str(model), model_repo=repo,
                    revision=revision, model_sha256=sha, model_size=size)
    ROOT.mkdir(exist_ok=True)
    output = ROOT / f'{args.model}.json'
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, indent=2)+'\n')
    os.replace(temporary,output)
    print('Installed:',output,'(active settings unchanged)',flush=True)


if __name__ == '__main__':
    main()
