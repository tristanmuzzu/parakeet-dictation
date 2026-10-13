"""Download pinned Whisper Turbo assets; leave active preferences unchanged."""
import argparse
import json
import os
from pathlib import Path
import sys
from setup_speech_model import ROOT,download

REPO='dropbox-dash/faster-whisper-large-v3-turbo'
REVISION='0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf'
FILES={
    'model.bin':('e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da',1617884929),
    'config.json':('b0253ea6c0d3bea6b1e19e91a02acfd3b53f4467362efcb5a3e6b16c9b3a9b7e',2263),
    'preprocessor_config.json':('7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711',340),
    'tokenizer.json':('297b13372ac43916285644fb9687add3cc62ee2a1adb60da3dc25cc94c1871fd',2710337),
    'vocabulary.json':('c69260f2ab26d659b7c398f9a2b2b48ed0df16c3b47d7326782fd9cba71690c1',1068114),
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-dir',type=Path,default=ROOT/'whisper-turbo')
    parser.add_argument('--python-packages',type=Path,help='Optional isolated package directory installed with this Python')
    args=parser.parse_args()
    if args.python_packages:sys.path.insert(0,str(args.python_packages.resolve()))
    try:import faster_whisper,ctranslate2
    except ImportError as exc:
        raise SystemExit('Install tools/requirements-whisper.txt with the app Python first.') from exc
    assets=args.assets_dir.resolve();assets.mkdir(parents=True,exist_ok=True)
    for name,(sha,size) in FILES.items():
        print('Downloading/verifying',name,flush=True)
        download(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}',assets/name,sha,size)
    manifest=dict(model='openai/whisper-large-v3-turbo',export_repo=REPO,revision=REVISION,
        directory=str(assets),files={n:dict(sha256=sha,size=size) for n,(sha,size) in FILES.items()},
        python_version=list(sys.version_info[:2]),packages=str(args.python_packages.resolve()) if args.python_packages else None,
        faster_whisper=faster_whisper.__version__,ctranslate2=ctranslate2.__version__)
    ROOT.mkdir(exist_ok=True);temp=ROOT/'whisper-turbo.json.tmp'
    temp.write_text(json.dumps(manifest,indent=2)+'\n');os.replace(temp,ROOT/'whisper-turbo.json')
    print('Installed whisper-turbo (active settings unchanged)',flush=True)


if __name__=='__main__':main()
