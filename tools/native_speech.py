"""Pinned NeMo-Speech C ABI adapter, loaded only inside the speech worker.

Owns native model/result handles. Audio remains in memory; no HTTP server or
second rewriting model. C declarations mirror the installed v0.1.0 asr.h.
"""
# Benchmark-only adapter: the running dictation app uses speech_models.py.
import ctypes as C
import json
import os
from pathlib import Path
import numpy as np


class Backend(C.Structure):
    _fields_ = [('size', C.c_size_t), ('gpu', C.c_int32)]


class Model(C.Structure):
    _fields_ = [('size', C.c_size_t), ('path', C.c_char_p), ('name', C.c_char_p)]


class Streaming(C.Structure):
    _fields_ = [('size',C.c_size_t),('chunk_size',C.c_float),
        ('ctc_left_padding',C.c_float),('ctc_right_padding',C.c_float),
        ('rnnt_right_context',C.c_int32)]


class Config(C.Structure):
    _fields_ = [('size', C.c_size_t)] + [(name, C.c_void_p) for name in
        ('backend','model','streaming','decoder','vad','endpointing','postproc','diar','batching')]


class Options(C.Structure):
    _fields_ = [('size',C.c_size_t),('request_id',C.c_char_p),('language_code',C.c_char_p),
        ('interim_results',C.c_bool),('enable_word_time_offsets',C.c_bool),
        ('enable_automatic_punctuation',C.c_bool),('verbatim_transcripts',C.c_bool),
        ('profanity_filter',C.c_bool),('stop_history_eou_ms',C.c_int32),
        ('speech_contexts',C.c_void_p),('speech_context_count',C.c_size_t),
        ('max_alternatives',C.c_int32),('enable_speaker_diarization',C.c_bool),
        ('max_speaker_count',C.c_int32)]


class NativeSpeech:
    def __init__(self, manifest, language='auto'):
        data = json.loads(Path(manifest).read_text())
        if data['runtime'] != 'NeMo-Speech.cpp 0.1.0':
            raise ValueError('Unsupported native speech runtime; rerun setup_speech_model.py')
        library, model_path = Path(data['library']), Path(data['model'])
        if not library.is_file() or not model_path.is_file():
            raise FileNotFoundError('Speech assets missing; run tools/setup_speech_model.py')
        if model_path.stat().st_size != data['model_size']:
            raise ValueError('Speech model is incomplete; rerun setup_speech_model.py')
        self._dll_directory = os.add_dll_directory(str(library.parent)) if os.name == 'nt' else None
        self._handle = C.c_void_p()
        self.library = C.CDLL(str(library))
        self.name = data['model_repo']
        self._language = language.encode('utf-8')
        declarations = {
            'create': ([C.POINTER(Config), C.POINTER(C.c_void_p)], C.c_int),
            'destroy': ([C.c_void_p], None),
            'recognition_options_default': ([], Options),
            'recognize_f32': ([C.c_void_p,C.POINTER(Options),C.POINTER(C.c_float),C.c_size_t,
                               C.c_int32,C.POINTER(C.c_void_p)],C.c_int),
            'result_alternative_count': ([C.c_void_p],C.c_size_t),
            'result_transcript': ([C.c_void_p,C.c_size_t],C.c_char_p),
            'result_destroy': ([C.c_void_p],None),
            'last_error': ([],C.c_char_p),
        }
        for name,(args,restype) in declarations.items():
            fn = getattr(self.library,'nemo_speech_asr_'+name)
            fn.argtypes,fn.restype = args,restype
            setattr(self,'_'+name,fn)
        backend = Backend(C.sizeof(Backend),-1)
        model = Model(C.sizeof(Model),str(model_path).encode('utf-8'),None)
        # Completed audio segments are already available: use the model's
        # largest supported context rather than its lowest-latency live mode.
        streaming = Streaming(C.sizeof(Streaming),.16,1.92,1.92,-1)
        config = Config()
        config.size = C.sizeof(Config)
        config.backend,config.model = C.addressof(backend),C.addressof(model)
        config.streaming = C.addressof(streaming)
        try:
            self._check(self._create(C.byref(config),C.byref(self._handle)))
        except Exception:
            self.close()
            raise

    def _check(self, status):
        if status:
            message = self._last_error()
            raise RuntimeError(message.decode('utf-8',errors='replace') if message else 'Native speech failed')

    def recognize(self, audio, sample_rate=16000):
        if not self._handle:
            raise RuntimeError('Speech model is closed')
        samples = np.asarray(audio,dtype=np.float32)
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError('Speech audio must be finite mono samples')
        if not 8000 <= sample_rate <= 96000:
            raise ValueError('Unsupported sample rate')
        if not samples.size:
            return ''
        samples = np.ascontiguousarray(samples)
        options = self._recognition_options_default()
        options.language_code = self._language
        options.enable_automatic_punctuation = True
        options.verbatim_transcripts = True
        options.profanity_filter = False
        result = C.c_void_p()
        try:
            self._check(self._recognize_f32(self._handle,C.byref(options),
                samples.ctypes.data_as(C.POINTER(C.c_float)),samples.size,sample_rate,C.byref(result)))
            if not result or not self._result_alternative_count(result):
                return ''
            text = self._result_transcript(result,0)
            return text.decode('utf-8').strip() if text else ''
        finally:
            if result:
                self._result_destroy(result)

    def close(self):
        if self._handle:
            self._destroy(self._handle)
            self._handle = C.c_void_p()
        if self._dll_directory is not None:
            self._dll_directory.close()
            self._dll_directory = None
