"""Pinned local native ASR in the existing worker; no rewriting or cloud calls.

The current Cohere trial uses an explicit English hint and Vulkan acceleration.
It must never silently fall back to the much slower CPU configuration.
"""
import json
import os
from pathlib import Path
import sys

PROFILES = {
    'cohere-transcribe-en': {
        'model': 'CohereLabs/cohere-transcribe-03-2026',
        'variant': 'cohere-transcribe-03-2026',
        'language': 'en',
        'quantization': 'Q8_0',
    },
}
RUNTIME = 'transcribe.cpp 0.2.4'


def model_assets(directory, choice):
    if choice not in PROFILES:
        raise ValueError('Unsupported native speech profile')
    try:
        data = json.loads((Path(directory)/'.local-speech'/(choice+'.json')).read_text())
        profile = PROFILES[choice]
        if data['runtime'] != RUNTIME or data['backend'] != 'vulkan':
            raise ValueError('Wrong native runtime or backend')
        if any(data[k] != value for k,value in profile.items()):
            raise ValueError('Wrong speech profile')
        paths = {}
        for role in ('model','library','binding'):
            item = data['files'][role]
            path = Path(item['path'])
            if not path.is_file() or path.stat().st_size != item['size']:
                raise ValueError('Missing or incomplete speech asset: '+role)
            paths[role] = path
        if paths['binding'].name != '__init__.py' or paths['binding'].parent.name != 'transcribe_cpp':
            raise ValueError('Unexpected Python binding path')
        return paths
    except (OSError,ValueError,TypeError,KeyError) as exc:
        raise RuntimeError('Install native speech assets with tools/setup_transcribe_speech.py, '
                           'or select parakeet and restart') from exc


class TranscribeSpeech:
    def __init__(self, directory, choice):
        if not sys.platform.startswith('linux'):
            raise RuntimeError('This native speech trial currently supports Linux only')
        self.model = self.session = None
        self.profile = PROFILES.get(choice)
        paths = model_assets(directory,choice)
        # This module is loaded only in the separate ASR worker. Bindings and
        # native runtime are installed together, isolated from global Python.
        sys.path.insert(0,str(paths['binding'].parents[1]))
        os.environ['TRANSCRIBE_LIBRARY'] = str(paths['library'])
        import transcribe_cpp as native
        if native.native_version() != '0.2.4':
            raise RuntimeError('Unexpected native speech runtime version')
        try:
            self.model = native.Model(paths['model'],backend='vulkan')
            if not self.model.backend.lower().startswith('vulkan'):
                raise RuntimeError('Vulkan speech acceleration is unavailable')
            if self.model.variant != self.profile['variant']:
                raise RuntimeError('Loaded speech model does not match selected profile')
            self.session = self.model.session(n_threads=4)
        except BaseException:
            self.close()
            raise

    def recognize(self, audio, sample_rate=16000):
        import numpy as np
        if self.session is None:
            raise RuntimeError('Speech model is closed')
        samples = np.asarray(audio,dtype=np.float32)
        if sample_rate != 16000 or samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError('Speech input must be finite mono audio at 16 kHz')
        if not samples.size:
            return ''
        # Native exceptions (including incomplete output) propagate to the
        # existing worker error path. Never return partial words as success.
        result = self.session.run(np.ascontiguousarray(samples),
                                  language=self.profile['language'],timestamps='none')
        if not isinstance(result.text,str):
            raise RuntimeError('Invalid native speech response')
        return result.text.strip()

    def close(self):
        session,self.session = self.session,None
        model,self.model = self.model,None
        try:
            if session is not None:session.close()
        finally:
            if model is not None:model.close()
