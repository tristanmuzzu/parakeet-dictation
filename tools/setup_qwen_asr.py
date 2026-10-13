"""Install pinned Qwen3-ASR Q8 speech weights and a local Linux CPU runtime.

No active preferences are changed. This is speech recognition, not prompt editing.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import tarfile
import tempfile
from setup_speech_model import ROOT, download, digest

RUNTIME_URL = 'https://github.com/ggml-org/llama.cpp/releases/download/b11146/llama-b11146-bin-ubuntu-x64.tar.gz'
RUNTIME_SHA = 'c150306eb16b5ab696f76a8bdf810c35fd98a24e82158742e6fa28f420ff8410'
MODELS = {
    'qwen-asr-0.6b': ('0.6B', '928ab958557df9aa2ef1c93e0e83c7ad0933fae2',
        ('bca259818b50ca7c4c05e9bdb35a5dc04fa039653a6d6f3f0f331f96f6aa1971',804749248),
        ('41a342b5e4c514e968cb756de6cd1b7be39eff43c44c57a2ef5fc6522e36603d',214392480)),
    'qwen-asr-1.7b': ('1.7B', '36a678687ba7d07a74ca70ccb0e36902e005fb80',
        ('58e22d0532d4eacaf034cfac17a6fed159f37c41390c710186783be439d1fc57',2165034944),
        ('46c1d533af3f354ceb37ce855dbceff7da7fa7cf1e6a523df3b13440bd164c0d',355709344)),
}


def install_runtime(archive, runtime):
    # Never overwrite a running executable or mapped library. Extract a fresh
    # version once; subsequent model installs verify and reuse it unchanged.
    with tempfile.TemporaryDirectory(prefix='.qwen-runtime-',dir=runtime.parent) as tmp:
        with tarfile.open(archive) as package:
            package.extractall(tmp,filter='data')
        source = Path(tmp)/'llama-b11146'
        target = runtime/'llama-b11146'
        if not target.exists():
            source.rename(target)
        else:
            for item in source.rglob('*'):
                existing = target/item.relative_to(source)
                if item.is_symlink():
                    valid = existing.is_symlink() and os.readlink(item)==os.readlink(existing)
                elif item.is_file():
                    valid = existing.is_file() and digest(item)==digest(existing)
                else:
                    valid = existing.is_dir()
                if not valid:
                    raise RuntimeError('Existing runtime differs; choose a fresh --assets-dir')
        return target/'llama-server'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=MODELS,default='qwen-asr-1.7b')
    parser.add_argument('--assets-dir',type=Path,default=ROOT/'qwen-asr')
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64','amd64'):
        raise SystemExit('This runtime installer currently supports Linux x64.')
    assets = args.assets_dir.resolve()
    assets.mkdir(parents=True,exist_ok=True)
    archive = assets/'llama-b11146.tar.gz'
    print('Verifying CPU runtime',flush=True)
    download(RUNTIME_URL,archive,RUNTIME_SHA)
    runtime = assets/'runtime'
    runtime.mkdir(exist_ok=True)
    server = install_runtime(archive,runtime)
    if not server.is_file(): raise RuntimeError('Runtime archive has unexpected layout')
    size,revision,weights,projector = MODELS[args.model]
    repo = f'ggml-org/Qwen3-ASR-{size}-GGUF'
    paths = {}
    for role,prefix,info in (('model','',weights),('projector','mmproj-',projector)):
        name = f'{prefix}Qwen3-ASR-{size}-Q8_0.gguf'
        path = assets/name
        print('Downloading/verifying',name,flush=True)
        download(f'https://huggingface.co/{repo}/resolve/{revision}/{name}',path,*info)
        paths[role] = dict(path=str(path),sha256=info[0],size=info[1])
    paths['server'] = dict(path=str(server),sha256=digest(server),size=server.stat().st_size)
    manifest = dict(model=f'Qwen/Qwen3-ASR-{size}',export_repo=repo,revision=revision,
                    runtime='llama.cpp b11146',runtime_sha256=RUNTIME_SHA,files=paths)
    ROOT.mkdir(exist_ok=True)
    temporary = ROOT/(args.model+'.json.tmp')
    temporary.write_text(json.dumps(manifest,indent=2)+'\n')
    os.replace(temporary,ROOT/(args.model+'.json'))
    print('Installed',args.model,'(active settings unchanged)',flush=True)


if __name__ == '__main__': main()
