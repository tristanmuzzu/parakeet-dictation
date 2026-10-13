"""Benchmark-only Whisper adapter; too slow in this workstation's CPU trial."""
import json
from pathlib import Path
import sys


def whisper_assets(directory):
    try:
        data=json.loads((Path(directory)/'.local-speech'/'whisper-turbo.json').read_text())
        if data['model']!='openai/whisper-large-v3-turbo':raise ValueError('Wrong model')
        assets=Path(data['directory'])
        for name in ('model.bin','config.json','tokenizer.json','preprocessor_config.json','vocabulary.json'):
            path=assets/name
            if not path.is_file() or path.stat().st_size!=data['files'][name]['size']:
                raise ValueError('Missing/incomplete Whisper asset: '+name)
        packages=data.get('packages')
        if packages and (data['python_version']!=list(sys.version_info[:2]) or not Path(packages).is_dir()):
            raise ValueError('Isolated packages do not match this Python')
        return assets,packages
    except (OSError,KeyError,ValueError,TypeError) as exc:
        raise RuntimeError('Install Whisper assets with tools/setup_whisper.py, or select parakeet and restart') from exc


class WhisperSpeech:
    def __init__(self,directory):
        assets,packages=whisper_assets(directory)
        if packages:sys.path.insert(0,packages)
        try:from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError('Whisper needs tools/requirements-whisper.txt installed with the app Python') from exc
        self.model=WhisperModel(str(assets),device='cpu',compute_type='int8',cpu_threads=4,
                                num_workers=1,local_files_only=True)

    def recognize(self,audio,sample_rate=16000):
        import numpy as np
        samples=np.asarray(audio,dtype=np.float32)
        if sample_rate!=16000 or samples.ndim!=1 or not np.isfinite(samples).all():
            raise ValueError('Whisper expects finite mono 16 kHz audio')
        if not samples.size:return ''
        segments,_=self.model.transcribe(samples,task='transcribe',language=None,
            beam_size=5,temperature=0,condition_on_previous_text=False,vad_filter=True,
            vad_parameters={'min_silence_duration_ms':500})
        return ' '.join(segment.text.strip() for segment in segments).strip()
