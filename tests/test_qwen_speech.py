"""Local ASR protocol, privacy, installer reuse and error-path checks."""
import io
import json
from pathlib import Path
import sys
import subprocess
import tarfile
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from qwen_speech import QwenSpeech,model_assets,transcript_from_response
from setup_qwen_asr import install_runtime


def response(text,finish='stop'):
    return {'choices':[{'finish_reason':finish,'message':{'content':text}}]}


class QwenTests(unittest.TestCase):
    def model(self):
        model=QwenSpeech.__new__(QwenSpeech)
        model.process=Mock();model.process.poll.return_value=None
        model._request=Mock(return_value=response('language German<asr_text>Keine Dateien löschen.'))
        return model

    def test_protocol_preserves_instruction_and_unicode(self):
        text='Do not change blue. No, make it red. Sprache bleibt Deutsch.'
        self.assertEqual(transcript_from_response(response('language English<asr_text>'+text)),text)
        self.assertEqual(transcript_from_response(response('language None<asr_text>')),'')
        self.assertEqual(transcript_from_response(response('The language is German.')),'The language is German.')

    def test_partial_and_protocol_errors_are_not_pasted(self):
        for data in (response('partial','length'),response(None),response('<|im_start|>garbage')):
            with self.assertRaises(RuntimeError): transcript_from_response(data)

    def test_audio_request_has_no_history_or_rewrite_prompt(self):
        model=self.model()
        self.assertEqual(model.recognize(np.ones(1600,dtype='float32')*.1),'Keine Dateien löschen.')
        method,path,body,timeout=model._request.call_args.args
        self.assertEqual(path,'/v1/chat/completions')
        self.assertEqual(body['messages'][0],{'role':'system','content':''})
        self.assertEqual(len(body['messages']),2)
        self.assertEqual(body['messages'][1]['content'][0]['type'],'input_audio')
        self.assertFalse(body['cache_prompt'])

    def test_invalid_audio_and_dead_runtime_fail_explicitly(self):
        model=self.model()
        for audio,rate in ((np.array([float('nan')]),16000),(np.zeros((2,2)),16000),(np.ones(50),8000)):
            with self.assertRaises(ValueError): model.recognize(audio,rate)
        model._request.assert_not_called()
        model.process.poll.return_value=1
        with self.assertRaises(RuntimeError): model.recognize(np.ones(1600))

    def test_http_error_closes_connection(self):
        model=self.model();model.port=12345;model.key='test-only'
        del model._request
        connection=Mock();connection.getresponse.return_value.status=500
        with patch('qwen_speech.http.client.HTTPConnection',return_value=connection) as ctor:
            with self.assertRaises(RuntimeError): model._request('GET','/health',None,1)
        ctor.assert_called_once_with('127.0.0.1',12345,timeout=1)
        connection.close.assert_called_once()

    def test_missing_assets_never_start_a_process(self):
        with tempfile.TemporaryDirectory() as tmp,patch('qwen_speech.subprocess.Popen') as launch:
            with self.assertRaisesRegex(RuntimeError,'Install selected Qwen'):
                QwenSpeech(tmp,'qwen-asr-0.6b')
            launch.assert_not_called()

    def test_close_reaps_only_owned_process(self):
        model=self.model();process=model.process
        model.close();model.close()
        process.terminate.assert_called_once();process.wait.assert_called_once_with(2)

    def test_runtime_reuse_does_not_overwrite_mapped_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/'runtime.tar';runtime=root/'runtime';runtime.mkdir()
            with tarfile.open(archive,'w') as tar:
                info=tarfile.TarInfo('llama-b11146/llama-server');info.size=3
                tar.addfile(info,io.BytesIO(b'bin'))
            server=install_runtime(archive,runtime)
            before=server.stat()
            self.assertEqual(install_runtime(archive,runtime),server)
            self.assertEqual(server.stat().st_ino,before.st_ino)
            self.assertEqual(server.stat().st_mtime_ns,before.st_mtime_ns)
            server.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'Existing runtime differs'):
                install_runtime(archive,runtime)
            self.assertEqual(server.read_bytes(),b'changed')

    @unittest.skipUnless(sys.platform.startswith('linux'),'Linux parent-death mechanism')
    def test_worker_death_reaps_native_runtime(self):
        # Exercise the exact exec wrapper with a harmless sleeper as runtime.
        # The readiness handshake proves the wrapper installed prctl first.
        launcher=str(ROOT/'qwen_speech.py')
        script='''import os,subprocess,sys,time
p=subprocess.Popen([sys.executable,sys.argv[1],'--exec-server',str(os.getpid()),
 sys.executable,'-c','import time;print("ready",flush=True);time.sleep(30)'],stdout=subprocess.PIPE,text=True)
assert p.stdout.readline().strip()=='ready'
print(p.pid,flush=True)
time.sleep(30)
'''
        parent=subprocess.Popen([sys.executable,'-c',script,launcher],stdout=subprocess.PIPE,text=True)
        child=None
        def running(pid):
            try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1][0]!='Z'
            except FileNotFoundError:return False
        try:
            child=int(parent.stdout.readline())
            self.assertTrue(running(child))
            parent.terminate();parent.wait(3)
            deadline=time.monotonic()+3
            while running(child) and time.monotonic()<deadline:time.sleep(.02)
            self.assertFalse(running(child),'Native runtime outlived its speech worker')
        finally:
            if parent.poll() is None:parent.kill();parent.wait(3)
            parent.stdout.close()
            if child and running(child):
                import os,signal
                os.kill(child,signal.SIGKILL)


if __name__=='__main__':unittest.main()
