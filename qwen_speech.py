"""Local Qwen3-ASR, owned by the existing speech worker. No text rewriting.

Audio stays in memory and travels only over authenticated loopback HTTP. On
Linux the runtime dies with its worker, including forced worker retirement.
"""
import base64
import http.client
import io
import json
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import sys
import time
import wave


def model_assets(directory, choice):
    if choice not in ('qwen-asr-0.6b','qwen-asr-1.7b'):
        raise ValueError('Unsupported Qwen speech model')
    try:
        data = json.loads((Path(directory)/'.local-speech'/(choice+'.json')).read_text())
        expected = 'Qwen/Qwen3-ASR-'+choice.removeprefix('qwen-asr-').upper()
        if data['model'] != expected or data['runtime'] != 'llama.cpp b11146':
            raise ValueError('Wrong model or runtime')
        paths = {}
        for role in ('server','model','projector'):
            item = data['files'][role]
            path = Path(item['path'])
            if not path.is_file() or path.stat().st_size != item['size']:
                raise ValueError('Missing/incomplete asset: '+role)
            paths[role] = path
        return paths
    except (OSError,KeyError,ValueError,TypeError) as exc:
        raise RuntimeError('Install selected Qwen speech assets with tools/setup_qwen_asr.py, '
                           'or select parakeet and restart') from exc


def transcript_from_response(data):
    choice = data['choices'][0]
    if choice.get('finish_reason') != 'stop':
        raise RuntimeError('Speech output was incomplete')
    text = choice['message']['content']
    if not isinstance(text,str): raise RuntimeError('Invalid speech output')
    # Qwen ASR emits a language header. Remove only that anchored protocol field.
    text = re.sub(r'^\s*language [^<\n]+<asr_text>', '', text, count=1)
    text = re.sub(r'^\s*<asr_text>', '', text, count=1)
    if '<asr_text>' in text or '<|' in text:
        raise RuntimeError('Unexpected speech protocol output')
    return text.strip()


class QwenSpeech:
    def __init__(self, directory, choice):
        if not sys.platform.startswith('linux'):
            raise RuntimeError('Qwen ASR runtime currently supports Linux only')
        paths = model_assets(directory,choice)
        self.process = None
        self.key = secrets.token_urlsafe(32)
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1',0))
            self.port = reservation.getsockname()[1]
        args = [str(paths['server']), '--model',str(paths['model']),
                '--mmproj',str(paths['projector']), '--host','127.0.0.1',
                '--port',str(self.port), '--threads','4', '--threads-batch','4',
                '--poll','0', '--poll-batch','0', '--ctx-size','4096',
                '--batch-size','256', '--ubatch-size','128', '--parallel','1',
                '--n-gpu-layers','0', '--no-mmproj-offload', '--no-webui', '--log-disable']
        env = {k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_','MTMD_'))}
        env['LLAMA_API_KEY'] = self.key
        try:
            # Small exec wrapper sets parent-death behavior without preexec_fn
            # (unsafe in a potentially multithreaded Python worker).
            self.process = subprocess.Popen([sys.executable, __file__, '--exec-server',
                str(os.getpid()), *args], env=env,stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            deadline = time.monotonic()+60
            while time.monotonic()<deadline:
                if self.process.poll() is not None: raise RuntimeError('Qwen speech runtime exited')
                try:
                    self._request('GET','/health',None,1)
                    break
                except (OSError,http.client.HTTPException,RuntimeError):
                    time.sleep(.1)
            else: raise TimeoutError('Qwen speech runtime did not become ready')
        except BaseException:
            self.close()
            raise

    def _request(self, method, path, body, timeout):
        connection = http.client.HTTPConnection('127.0.0.1',self.port,timeout=timeout)
        try:
            connection.request(method,path,None if body is None else json.dumps(body),
                {'Content-Type':'application/json','Authorization':'Bearer '+self.key})
            response = connection.getresponse()
            payload = response.read()
            if response.status != 200: raise RuntimeError('Local ASR request failed: '+str(response.status))
            return json.loads(payload)
        finally:
            connection.close()

    def recognize(self, audio, sample_rate=16000):
        import numpy as np
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError('Speech runtime is closed')
        samples = np.asarray(audio,dtype=np.float32)
        if sample_rate != 16000 or samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError('Qwen ASR expects finite mono 16 kHz audio')
        if not samples.size: return ''
        with io.BytesIO() as buffer:
            with wave.open(buffer,'wb') as wav:
                wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(sample_rate)
                wav.writeframes((np.clip(samples,-1,1)*32767).astype('<i2').tobytes())
            encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
        response = self._request('POST','/v1/chat/completions',{
            'messages':[{'role':'system','content':''}, {'role':'user','content':[
                {'type':'input_audio','input_audio':{'data':encoded,'format':'wav'}}]}],
            'temperature':0,'max_tokens':1024,'cache_prompt':False},80)
        return transcript_from_response(response)

    def close(self):
        process,self.process = self.process,None
        if process is not None and process.poll() is None:
            process.terminate()
            try: process.wait(2)
            except subprocess.TimeoutExpired:
                process.kill();process.wait(2)


if __name__ == '__main__':
    if len(sys.argv)<5 or sys.argv[1]!='--exec-server': raise SystemExit('Internal runtime launcher')
    import ctypes
    import signal
    parent = int(sys.argv[2])
    if ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGKILL,0,0,0) != 0:
        raise OSError(ctypes.get_errno(),'Cannot bind speech runtime lifetime to worker')
    if os.getppid()!=parent: raise SystemExit('Speech worker exited before launch')
    os.execv(sys.argv[3],sys.argv[3:])
