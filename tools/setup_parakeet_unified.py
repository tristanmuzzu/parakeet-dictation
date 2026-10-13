"""Install the pinned INT8 Parakeet Unified English export for onnx-asr.

Model weights only: no remote Python code is executed. Protobuf is used solely
at setup time to read SentencePiece's vocabulary; inference needs no new package.
"""
import argparse
import json
import os
from pathlib import Path
from setup_speech_model import download, ROOT, digest

REPO = 'eschmidbauer/parakeet-unified-en-0.6b-onnx'
REVISION = '7a16ff98b72e5b0beba6f182a5357c704d977b15'
FILES = (
    ('encoder.int8.onnx','encoder-model.int8.onnx','8c430baee5a815b4ceee861f5f51b5b44db5836623ab2bfe9d0386e57b3f25ab',654040125),
    ('decoder_joint.int8.onnx','decoder_joint-model.int8.onnx','26a9f0ffb682cf2a712ac86e2e11b6da932f9860f70dc996841cba6392010b55',8995073),
    ('tokenizer.model','tokenizer.model','07d4e5a63840a53ab2d4d106d2874768143fb3fbdd47938b3910d2da05bfb0a9',251056),
)


def vocabulary(data):
    # Read the two fields needed from SentencePiece ModelProto. Protobuf safely
    # skips its remaining trainer/normalizer metadata; no custom binary parser.
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    schema=descriptor_pb2.FileDescriptorProto(name='dictation_sentencepiece.proto',package='dictation')
    piece=schema.message_type.add(name='Piece')
    piece.field.add(name='piece',number=1,type=9,label=1)
    model=schema.message_type.add(name='Model')
    model.field.add(name='pieces',number=1,type=11,label=3,type_name='.dictation.Piece')
    pool=descriptor_pool.DescriptorPool();pool.Add(schema)
    proto=message_factory.GetMessageClass(pool.FindMessageTypeByName('dictation.Model'))()
    proto.ParseFromString(data)
    tokens=[p.piece for p in proto.pieces]
    if len(tokens)!=1024 or any(not t or any(c in t for c in (' ','\n','\r')) for t in tokens):
        raise ValueError('Unexpected Parakeet Unified vocabulary')
    return ''.join(f'{token} {i}\n' for i,token in enumerate(tokens))+'<blk> 1024\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-dir',type=Path,default=ROOT/'parakeet-unified')
    args=parser.parse_args()
    try:
        from google.protobuf import message_factory
        if not hasattr(message_factory, 'GetMessageClass'):
            raise ImportError('Protobuf is too old')
    except ImportError as exc:
        raise SystemExit('Setup needs protobuf>=4.25; install tools/requirements-speech-setup.txt with this Python.') from exc
    assets=args.assets_dir.resolve();assets.mkdir(parents=True,exist_ok=True)
    for source,target,sha,size in FILES:
        print('Downloading/verifying',source,flush=True)
        download(f'https://huggingface.co/{REPO}/resolve/{REVISION}/onnx_int8/{source}',assets/target,sha,size)
    (assets/'vocab.txt').write_text(vocabulary((assets/'tokenizer.model').read_bytes()),encoding='utf-8')
    (assets/'config.json').write_text(json.dumps(dict(model_type='nemo-conformer-rnnt',features_size=128,subsampling_factor=8),indent=2)+'\n')
    manifest=dict(model='nvidia/parakeet-unified-en-0.6b',export_repo=REPO,revision=REVISION,
        directory=str(assets),files={name:dict(sha256=digest(assets/name),size=(assets/name).stat().st_size)
        for name in [f[1] for f in FILES]+['vocab.txt','config.json']})
    ROOT.mkdir(exist_ok=True)
    temp=ROOT/'parakeet-unified.json.tmp';temp.write_text(json.dumps(manifest,indent=2)+'\n')
    os.replace(temp,ROOT/'parakeet-unified.json')
    print('Installed Parakeet Unified INT8 (active settings unchanged)',flush=True)


if __name__=='__main__':main()
