"""Install pinned Linux Vulkan speech assets without changing live preferences."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import tarfile
import tempfile
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from transcribe_speech import PROFILES,RUNTIME
from setup_speech_model import ROOT,download,digest

RELEASE = 'https://github.com/handy-computer/transcribe.cpp/releases/download/v0.2.4/'
ARCHIVE = ('transcribe-native-0.2.4-linux-x86_64-cpu-vulkan.tar.gz',
           '28b22a523a25b41d59ff91147b6f79f35330663c92c22e483d15d6f4dc0cfc9a',21607373)
WHEEL = ('transcribe_cpp-0.2.4-py3-none-any.whl',
         '9ba8d54ada60ffb2f182dd7ae6fd84f478f90ec1426d7b66bfb2396ff3bfc261',34910)
MODELS = {
    'cohere-transcribe-en': ('handy-computer/cohere-transcribe-03-2026-gguf',
        'ab667acedcb5d8d56837ef3ed3568c3ad50dde47',
        'cohere-transcribe-03-2026-Q8_0.gguf',
        '931916663432fd895423a4291a8400221802b288967ca2d435fc5e3141c9e71e',2410655232),
}


def reuse_tree(source,target):
    """Install once; never overwrite libraries a worker may have mapped."""
    if not target.exists():
        source.rename(target)
        return
    for item in source.rglob('*'):
        existing = target/item.relative_to(source)
        if item.is_symlink():
            valid = existing.is_symlink() and os.readlink(existing)==os.readlink(item)
        elif item.is_file():
            valid = existing.is_file() and digest(existing)==digest(item)
        else:
            valid = existing.is_dir()
        if not valid:
            raise RuntimeError('Installed runtime differs; choose a fresh --assets-dir')


def install_runtime(archive,wheel,assets):
    runtime=assets/'runtime';runtime.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.native-speech-',dir=assets) as td:
        tmp=Path(td)
        with tarfile.open(archive) as package:package.extractall(tmp/'native',filter='data')
        name='transcribe-native-linux-x86_64-cpu-vulkan'
        reuse_tree(tmp/'native'/name,runtime/name)
        with zipfile.ZipFile(wheel) as package:
            target=tmp/'python'
            for info in package.infolist():
                if not (target/info.filename).resolve().is_relative_to(target.resolve()):
                    raise RuntimeError('Invalid binding archive path')
            package.extractall(target)
        reuse_tree(tmp/'python',assets/'python')
    return runtime/name/'libtranscribe.so',assets/'python/transcribe_cpp/__init__.py'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=MODELS,default='cohere-transcribe-en')
    p.add_argument('--assets-dir',type=Path,default=ROOT/'transcribe')
    args=p.parse_args()
    if platform.system()!='Linux' or platform.machine().lower() not in ('x86_64','amd64'):
        raise SystemExit('This installer currently supports Linux x64 only')
    assets=args.assets_dir.resolve();assets.mkdir(parents=True,exist_ok=True)
    for name,sha,size in (ARCHIVE,WHEEL):download(RELEASE+name,assets/name,sha,size)
    library,binding=install_runtime(assets/ARCHIVE[0],assets/WHEEL[0],assets)
    repo,revision,name,sha,size=MODELS[args.model]
    model=assets/name
    download('https://huggingface.co/'+repo+'/resolve/'+revision+'/'+name,model,sha,size)
    paths={'model':model,'library':library,'binding':binding}
    manifest=dict(PROFILES[args.model],runtime=RUNTIME,backend='vulkan',export_repo=repo,
                  revision=revision,files={k:dict(path=str(v),size=v.stat().st_size,sha256=digest(v))
                                           for k,v in paths.items()})
    ROOT.mkdir(exist_ok=True)
    target=ROOT/(args.model+'.json');temporary=target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest,indent=2)+'\n');os.replace(temporary,target)
    print('Installed',args.model,'(active preferences unchanged)',flush=True)

if __name__=='__main__':main()
