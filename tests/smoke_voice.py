"""Bounded real-model regression with generated, non-private speech.

Requires the already cached model, espeak-ng and ffmpeg. Never opens the mic or
pastes. Run under a memory/CPU limited scope on the workstation.
"""
import multiprocessing as mp
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import wave
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ["HF_HUB_OFFLINE"]="1"
import dictation as d


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=d.SPEECH_MODELS, default='parakeet')
    parser.add_argument('--context-seconds',type=float,default=None)
    args=parser.parse_args()
    context=args.context_seconds if args.context_seconds is not None else (4 if args.model=='cohere-transcribe-en' else 0)
    with tempfile.TemporaryDirectory(prefix="parakeet-voice-") as td:
        original=Path(td)/"source.wav";wav=Path(td)/"speech.wav"
        subprocess.run(["espeak-ng","-s","145","-w",str(original),
                        "Please simplify the settings and keep the important controls within reach. Keep the final words."],check=True,timeout=10)
        subprocess.run(["ffmpeg","-loglevel","error","-i",str(original),"-ar","16000","-ac","1",str(wav)],check=True,timeout=10)
        with wave.open(str(wav)) as f:audio=np.frombuffer(f.readframes(f.getnframes()),dtype="int16").astype("float32")/32768
        parent,child=mp.Pipe();worker=mp.Process(target=d.asr_worker,args=(child,args.model),daemon=True)
        worker.start();child.close()
        try:
            assert parent.poll(90),"Worker readiness deadline exceeded"
            kind,payload=parent.recv();assert kind=="ready",(kind,payload)
            print("cached model ready",flush=True)
            for name,take in (("short",audio),("long",np.concatenate([audio,np.zeros(5000),audio,np.zeros(5000),audio]))):
                sess=d.AsrSession(parent,threading.Lock(), context_seconds=context)
                sess.frames=[take[i:i+2048] for i in range(0,len(take),2048)]
                sess.stop_feeding();sess.close();sess.join_consumer(90)
                assert not sess._consumer.is_alive(),"Consumer deadline exceeded"
                assert not sess.errors,sess.errors
                text=" ".join(sess.ordered_texts())
                print(name,round(len(take)/16000,2),"seconds:",repr(text),flush=True)
                assert "settings" in text.lower(),text
                assert text.lower().rstrip(" .!?").endswith("final words"),text
            # Reproduce a final 1.6-second tail after a hard boundary. It must
            # be decoded with preceding context, including a corrected final word.
            sess=d.AsrSession(parent,threading.Lock(), context_seconds=context)
            take=np.tile(audio,3)
            boundary=len(take)-int(1.6*16000)
            sess.segmenter=d.Segmenter(min_seg=100,max_seg=boundary/16000,boundary_window=0)
            sess.frames=[take[i:i+2048] for i in range(0,len(take),2048)]
            sess._drain_frames()
            stopped=time.monotonic();sess.stop_feeding();sess.close();sess.join_consumer(30)
            text=" ".join(sess.ordered_texts())
            assert not sess._consumer.is_alive() and not sess.errors,sess.errors
            assert text.lower().rstrip(" .!?").endswith("final words"),text
            assert text.lower().count("settings")==3,text
            assert text.lower().count("final words")==3,text
            print("hard-boundary short ending retained; finalization:",round(time.monotonic()-stopped,3),"seconds",flush=True)
            # Feed at microphone speed: prove work completes while speaking,
            # then measure only the remaining work after the second hotkey.
            take=np.concatenate([audio,np.zeros(16000,dtype="float32"),audio])
            sess=d.AsrSession(parent,threading.Lock(), context_seconds=context);sess.start_feeder()
            started=time.monotonic()
            for i in range(0,len(take),2048):
                block=take[i:i+2048]
                with sess.frames_lock:sess.frames.append(block)
                time.sleep(max(0,started+(i+len(block))/16000-time.monotonic()))
            completed_before_stop=len(sess.results)
            stopped=time.monotonic()
            sess.stop_feeding();sess.close();sess.join_consumer(30)
            elapsed=time.monotonic()-stopped
            assert not sess._consumer.is_alive(),"Streaming consumer deadline exceeded"
            assert not sess.errors,sess.errors
            text=" ".join(sess.ordered_texts())
            assert completed_before_stop>=1,"No transcription completed during speech"
            assert text.lower().rstrip(" .!?").endswith("final words"),text
            print("streamed",round(len(take)/16000,2),"seconds; chunks before stop:",completed_before_stop,
                  "; stop-to-text:",round(elapsed,3),"seconds; text:",repr(text),flush=True)
            print("VOICE_SMOKE_OK",flush=True)
        finally:
            if worker.is_alive():
                try:parent.send(("quit",None))
                except (OSError,EOFError):pass
                worker.join(3)
                if worker.is_alive():worker.terminate();worker.join(3)
            parent.close()

if __name__=="__main__":
    mp.freeze_support();main()
