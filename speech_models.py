"""Load one selected speech model inside the existing ASR worker.

The baseline retains its original INT8/fp32 fallback. The Unified trial requires
installed, pinned local assets and fails explicitly when they are unavailable.
"""
import json
import logging
from pathlib import Path


def unified_directory(directory):
    manifest = Path(directory)/'.local-speech'/'parakeet-unified.json'
    try:
        data = json.loads(manifest.read_text())
        assets = Path(data['directory'])
        if data['model'] != 'nvidia/parakeet-unified-en-0.6b':
            raise ValueError('Wrong speech model manifest')
        for name in ('encoder-model.int8.onnx','decoder_joint-model.int8.onnx','vocab.txt','config.json'):
            path = assets/name
            if not path.is_file() or path.stat().st_size != data['files'][name]['size']:
                raise ValueError('Missing or incomplete speech asset: '+name)
        return assets
    except (OSError,KeyError,ValueError,TypeError) as exc:
        raise RuntimeError('Install Parakeet Unified with tools/setup_parakeet_unified.py, '
                           'or set speech_model to parakeet and restart') from exc


def load_speech_model(choice, directory, is_linux=True):
    if choice == 'cohere-transcribe-en':
        from transcribe_speech import TranscribeSpeech
        return TranscribeSpeech(directory,choice)
    if choice == 'qwen-asr-1.7b':
        from qwen_speech import QwenSpeech
        return QwenSpeech(directory, choice)
    if choice not in ('parakeet','parakeet-unified'):
        raise ValueError('Unsupported speech model')
    assets = unified_directory(directory) if choice == 'parakeet-unified' else None
    import onnx_asr
    options = {}
    if is_linux:
        import onnxruntime as ort
        session = ort.SessionOptions()
        session.intra_op_num_threads = 4
        session.inter_op_num_threads = 1
        session.add_session_config_entry('session.intra_op.allow_spinning','0')
        options['sess_options'] = session
    if assets is not None:
        return onnx_asr.load_model('nemo-conformer-rnnt',assets,quantization='int8',**options)
    try:
        return onnx_asr.load_model('nemo-parakeet-tdt-0.6b-v3',quantization='int8',**options)
    except Exception:
        logging.getLogger('dictation').exception('worker: int8 load failed, falling back to fp32')
        return onnx_asr.load_model('nemo-parakeet-tdt-0.6b-v3',**options)
