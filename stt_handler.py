import time
import queue
import collections
import numpy as np
import torch
import psutil
import os
import config

def run_stt_process(audio_queue, result_queue, command_queue):
    """
    Standalone process for Speech-to-Text.
    Handles VAD (Voice Activity Detection), Audio Buffering, and Transcription.
    Running this isolated prevents the Discord bot from freezing during heavy inference.
    """
    print(f"STT Process started. PID: {os.getpid()}")
    
    print(f"STT Process started. PID: {os.getpid()}")
    
    # --- Model Initialization ---
    # Models must be loaded within this process to avoid CUDA context issues with multiprocessing.
    print("Loading Faster-Whisper model...")
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(config.STT_MODEL_ID, device=config.STT_DEVICE, compute_type=config.STT_COMPUTE_TYPE)
        print("Faster-Whisper loaded on GPU.")
    except Exception as e:
        print(f"Failed to load GPU model: {e}. Fallback to CPU base.")
        from faster_whisper import WhisperModel
        model = WhisperModel("base", device="cpu", compute_type="int8")

    print("Loading Silero VAD...")
    try:
        # Silero VAD is highly optimized for speech detection
        vad_model, utils = torch.hub.load(repo_or_dir=config.VAD_REPO_OR_DIR,
                                          model=config.VAD_MODEL,
                                          force_reload=False,
                                          trust_repo=True)
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
        print("Silero VAD loaded.")
    except Exception as e:
        print(f"Error loading Silero VAD: {e}")
        return

    # --- State Management ---
    user_buffers = {}       # Incoming raw audio stream
    user_speech_buffers = {} # Accumulated speech segments
    user_ring_buffers = {}   # Pre-speech context (Ring Buffer)
    user_last_activity = {}  # Timestamp for cleanup
    
    # VAD State per user
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
                user_ring_buffers[user_id] = collections.deque(maxlen=config.RING_BUFFER_SIZE)
                user_vad_iterators[user_id] = VADIterator(vad_model)
            
            user_buffers[user_id].extend(pcm_data)
            
            # Process audio in chunks of 512 samples (32ms)
            while len(user_buffers[user_id]) >= config.FRAME_SIZE_BYTES:
                frame = user_buffers[user_id][:config.FRAME_SIZE_BYTES]
                del user_buffers[user_id][:config.FRAME_SIZE_BYTES]
                
                # Prepare frame for Silero (float32, normalized)
                frame_np = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
                frame_tensor = torch.from_numpy(frame_np)
                
                # Get speech probability
                speech_dict = user_vad_iterators[user_id](frame_tensor, return_seconds=True)
                
                is_speech = False
                # Check if VAD triggered
                if user_vad_iterators[user_id].triggered:
                    is_speech = True
                
                if is_speech:
                    # Speech detected.
                    # If this is the start of a new speech segment, prepend the ring buffer context.
                    if len(user_speech_buffers[user_id]) == 0:
                         for prev_frame in user_ring_buffers[user_id]:
                             user_speech_buffers[user_id].extend(prev_frame)
                         user_ring_buffers[user_id].clear()
                    
                    user_speech_buffers[user_id].extend(frame)
                else:
                    # Silence detected.
                    if len(user_speech_buffers[user_id]) > 0:
                        # End of speech segment -> Transcribe
                        audio_to_transcribe = user_speech_buffers[user_id][:]
                        user_speech_buffers[user_id] = bytearray()
                        
                        transcribe_and_send(model, user_id, audio_to_transcribe, result_queue)
                    
                    # Keep recent frames in ring buffer for context
                    user_ring_buffers[user_id].append(frame)

        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error in STT loop: {e}")

        # 3. Cleanup Inactive Users
        current_time = time.time()
        users_to_remove = []
        for uid, last_time in user_last_activity.items():
            if current_time - last_time > config.USER_TIMEOUT_SECONDS:
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
        segments, info = model.transcribe(data_f32, language=config.STT_LANGUAGE, beam_size=config.STT_BEAM_SIZE)
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
