"""Selection, failure recovery and pinned installer checks; no downloads/ASR."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
from app_store import AppStore
from speech_models import load_speech_model, unified_directory
from setup_speech_model import download
from setup_parakeet_unified import vocabulary


class SpeechModels(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def install_fixture(self):
        assets = self.root/'assets'
        assets.mkdir()
        files = {}
        for name in ('encoder-model.int8.onnx','decoder_joint-model.int8.onnx','vocab.txt','config.json'):
            (assets/name).write_bytes(b'test')
            files[name] = {'size': 4}
        manifest = self.root/'.local-speech'
        manifest.mkdir()
        (manifest/'parakeet-unified.json').write_text(json.dumps({
            'model':'nvidia/parakeet-unified-en-0.6b', 'directory':str(assets), 'files':files}))
        return assets

    def test_selection_persists_and_rollback_keeps_other_preferences(self):
        store = AppStore(self.root)
        store.update('input_device','test-mic')
        store.update('speech_model','parakeet-unified')
        store = AppStore(self.root)
        self.assertEqual(store.settings['speech_model'],'parakeet-unified')
        self.assertFalse(store.settings['light_edit'])
        store.update('speech_model','parakeet')
        self.assertEqual(AppStore(self.root).settings['input_device'],'test-mic')
        with self.assertRaises(ValueError): store.update('speech_model','unknown')
        (self.root/'settings.json').write_text('{"speech_model":"unknown"}')
        self.assertEqual(AppStore(self.root).settings['speech_model'],'parakeet')

    def test_missing_or_partial_assets_fail_before_loading(self):
        loader = Mock()
        with patch.dict(sys.modules, {'onnx_asr':loader}):
            with self.assertRaisesRegex(RuntimeError,'Install Parakeet Unified'):
                load_speech_model('parakeet-unified',self.root)
            assets = self.install_fixture()
            (assets/'encoder-model.int8.onnx').write_bytes(b'partial')
            with self.assertRaises(RuntimeError): unified_directory(self.root)
        loader.load_model.assert_not_called()

    def test_unified_uses_local_rnnt_and_does_not_silently_fallback(self):
        assets = self.install_fixture()
        loader = Mock()
        with patch.dict(sys.modules, {'onnx_asr':loader}):
            self.assertIs(load_speech_model('parakeet-unified',self.root,False),loader.load_model.return_value)
            loader.load_model.assert_called_once_with('nemo-conformer-rnnt',assets,quantization='int8')
            loader.load_model.reset_mock()
            loader.load_model.side_effect = RuntimeError('invalid graph')
            with self.assertRaisesRegex(RuntimeError,'invalid graph'):
                load_speech_model('parakeet-unified',self.root,False)
            self.assertEqual(loader.load_model.call_count,1)

    def test_baseline_retains_float_fallback(self):
        loader = Mock()
        loader.load_model.side_effect = [RuntimeError('int8 unavailable'),'baseline']
        with patch.dict(sys.modules, {'onnx_asr':loader}):
            self.assertEqual(load_speech_model('parakeet',self.root,False),'baseline')
        self.assertEqual(loader.load_model.call_args.kwargs,{})

    def test_candidates_load_only_the_selected_backend(self):
        for key,module,cls in (('qwen-asr-1.7b','qwen_speech','QwenSpeech'),
                               ('cohere-transcribe-en','transcribe_speech','TranscribeSpeech')):
            store=AppStore(self.root)
            store.update('speech_model',key)
            self.assertEqual(AppStore(self.root).settings['speech_model'],key)
            with patch(module+'.'+cls) as selected,patch.dict(sys.modules,{'onnx_asr':None}):
                self.assertIs(load_speech_model(key,self.root),selected.return_value)
                selected.assert_called_once()

    def test_worker_reports_load_error_without_ready(self):
        import dictation
        conn = Mock()
        with patch('speech_models.load_speech_model',side_effect=RuntimeError('missing assets')):
            dictation.asr_worker(conn,'parakeet-unified')
        conn.send.assert_called_once_with(('load_error','missing assets'))

    def test_download_failure_preserves_previous_file(self):
        target = self.root/'model'
        target.write_bytes(b'previous')
        with patch('urllib.request.urlopen',return_value=io.BytesIO(b'broken')):
            with self.assertRaisesRegex(RuntimeError,'Checksum'):
                download('https://example.invalid/model',target,hashlib.sha256(b'correct').hexdigest(),6)
        self.assertEqual(target.read_bytes(),b'previous')
        self.assertFalse(target.with_suffix('.part').exists())

    def test_tokenizer_conversion_preserves_unicode_and_blank(self):
        # Minimal SentencePiece wire data with 1024 repeated Piece messages.
        tokens = ['▁hello'] + ['token'+str(i) for i in range(1023)]
        data = b''
        for token in tokens:
            encoded = token.encode()
            piece = b'\x0a'+bytes([len(encoded)])+encoded
            data += b'\x0a'+bytes([len(piece)])+piece
        result = vocabulary(data)
        self.assertTrue(result.startswith('▁hello 0\n'))
        self.assertTrue(result.endswith('<blk> 1024\n'))
        with self.assertRaises(ValueError): vocabulary(b'')


if __name__ == '__main__': unittest.main()
