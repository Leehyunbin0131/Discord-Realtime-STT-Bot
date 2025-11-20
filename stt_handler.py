import time
import queue
import collections
import numpy as np
import torch
import psutil
import os

def run_stt_process(audio_queue, result_queue, command_queue):
    """
    This function runs in a separate process.
    It handles VAD, buffering, and transcription.
    """
    print(f"STT Process started. PID: {os.getpid()}")
    
    # 1. Load Models (MUST be done inside the process)
    print("Loading Faster-Whisper model...")
    try:
        from faster_whisper import WhisperModel
        # Use 'large-v3-turbo' as requested previously, or fallback to base if too heavy
        model_id = "deepdml/faster-whisper-large-v3-turbo-ct2"
        model = WhisperModel(model_id, device="cuda", compute_type="float16")
        print("Faster-Whisper loaded on GPU.")
    except Exception as e:
        print(f"Failed to load GPU model: {e}. Fallback to CPU base.")
        from faster_whisper import WhisperModel
        model = WhisperModel("base", device="cpu", compute_type="int8")

    print("Loading Silero VAD...")
    try:
        # Load Silero VAD from torch hub
        vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                          model='silero_vad',
                                          force_reload=False,
                                          trust_repo=True)
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
        print("Silero VAD loaded.")
    except Exception as e:
        print(f"Error loading Silero VAD: {e}")
        return

    # 2. State Management
    user_buffers = {} # user_id -> bytearray (incoming raw stream)
    user_speech_buffers = {} # user_id -> bytearray (accumulated speech)
    user_ring_buffers = {} # user_id -> deque (pre-speech context)
    user_last_activity = {} # user_id -> timestamp
    
    # Constants
    SAMPLE_RATE = 16000
    # Silero VAD requires 512, 1024, 1536 samples for 16kHz
    # 512 samples / 16000Hz = 0.032s = 32ms
    FRAME_SIZE_SAMPLES = 512
    FRAME_SIZE_BYTES = FRAME_SIZE_SAMPLES * 2 # 1024 bytes
    RING_BUFFER_SIZE = 10 # ~320ms context
    
    # Cleanup Config
    USER_TIMEOUT_SECONDS = 60
    
    # VAD Iterator per user
    user_vad_iterators = {} 

    print("STT Process Ready.")

    while True:
        # 1. Check Commands
        try:
            while not command_queue.empty():
                cmd, data = command_queue.get_nowait()
                if cmd == "LEAVE":
                    user_id = data
                    if user_id in user_buffers:
                        del user_buffers[user_id]
                    if user_id in user_speech_buffers:
                        del user_speech_buffers[user_id]
                    if user_id in user_ring_buffers:
                        del user_ring_buffers[user_id]
                    if user_id in user_last_activity:
                        del user_last_activity[user_id]
                    if user_id in user_vad_iterators:
                        del user_vad_iterators[user_id]
                    print(f"Cleaned up user {user_id}")
        except Exception:
            pass

        # 2. Process Audio
        try:
            # Non-blocking get with timeout to allow cleanup loop to run
            user_id, pcm_data = audio_queue.get(timeout=0.1)
            
            # Update Activity
            user_last_activity[user_id] = time.time()
            
            # Initialize User State
            if user_id not in user_buffers:
                user_buffers[user_id] = bytearray()
                user_speech_buffers[user_id] = bytearray()
                user_ring_buffers[user_id] = collections.deque(maxlen=RING_BUFFER_SIZE)
                user_vad_iterators[user_id] = VADIterator(vad_model)
            
            user_buffers[user_id].extend(pcm_data)
            
            # Process in 512 sample chunks (1024 bytes)
            while len(user_buffers[user_id]) >= FRAME_SIZE_BYTES:
                frame = user_buffers[user_id][:FRAME_SIZE_BYTES]
                del user_buffers[user_id][:FRAME_SIZE_BYTES]
                
                # Convert frame to tensor for Silero
                # Silero expects float32 tensor, normalized to [-1, 1]
                frame_np = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
                frame_tensor = torch.from_numpy(frame_np)
                
                # Get speech probability
                # VADIterator handles state and thresholding internally
                speech_dict = user_vad_iterators[user_id](frame_tensor, return_seconds=True)
                
                is_speech = False
                if speech_dict:
                    if 'start' in speech_dict:
                        # Speech started
                        pass
                    if 'end' in speech_dict:
                        # Speech ended
                        pass
                
                # Check current state
                if user_vad_iterators[user_id].triggered:
                    is_speech = True
                
                # Logic:
                if is_speech:
                    # If we just started speaking (buffer empty), add context
                    if len(user_speech_buffers[user_id]) == 0:
                         for prev_frame in user_ring_buffers[user_id]:
                             user_speech_buffers[user_id].extend(prev_frame)
                         user_ring_buffers[user_id].clear()
                    
                    user_speech_buffers[user_id].extend(frame)
                else:
                    # Not speaking
                    if len(user_speech_buffers[user_id]) > 0:
                        # Transcribe!
                        audio_to_transcribe = user_speech_buffers[user_id][:]
                        user_speech_buffers[user_id] = bytearray()
                        
                        # Transcribe function
                        transcribe_and_send(model, user_id, audio_to_transcribe, result_queue)
                    
                    # Add to ring buffer
                    user_ring_buffers[user_id].append(frame)

        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error in STT loop: {e}")

        # 3. Cleanup Inactive Users
        current_time = time.time()
        users_to_remove = []
        for uid, last_time in user_last_activity.items():
            if current_time - last_time > USER_TIMEOUT_SECONDS:
                users_to_remove.append(uid)
        
        for uid in users_to_remove:
            print(f"User {uid} timed out. Cleaning up.")
            del user_buffers[uid]
            del user_speech_buffers[uid]
            del user_ring_buffers[uid]
            del user_last_activity[uid]
            del user_vad_iterators[uid]

def transcribe_and_send(model, user_id, audio_data, result_queue):
    if len(audio_data) < 3200: # Ignore very short audio (< 0.1s)
        return

    data_s16 = np.frombuffer(audio_data, dtype=np.int16)
    data_f32 = data_s16.astype(np.float32) / 32768.0
    
    start_time = time.time()
    try:
        segments, info = model.transcribe(data_f32, language="ko", beam_size=1)
        text = ""
        for segment in segments:
            text += segment.text
        
        end_time = time.time()
        duration = end_time - start_time
        
        if text.strip():
            result_queue.put({
                "user_id": user_id,
                "text": text.strip(),
                "latency": f"{duration:.3f}s"
            })
    except Exception as e:
        print(f"Transcription error: {e}")
