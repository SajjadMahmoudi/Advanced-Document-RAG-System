from kokoro import KPipeline
import numpy as np
import io
import wave
import torch

class StreamingTTSProcessor:
    def __init__(self, lang_code='b'):
        """
        Initialize TTS processor
        lang_code: 'a' = American English, 'b' = British English, 
                   'e' = Spanish, 'f' = French, 'h' = Hindi, 
                   'i' = Italian, 'p' = Portuguese, 'j' = Japanese, 
                   'z' = Mandarin Chinese
        """
        self.pipeline = KPipeline(lang_code=lang_code)
        self.sample_rate = 24000
        
    def stream_speech(self, text, voice='af_heart', speed=1.0, chunk_size_ms=None):
        """
        Stream speech audio from Kokoro TTS.

        Args:
            text (str): Input text to synthesize.
            voice (str): Voice model.
            speed (float): Speech speed.
            chunk_size_ms (int or None):
                - None = Use Kokoro’s native chunking (sentence-based)
                - int (ms) = Force fixed-size streaming chunks (e.g., 200ms)

        Yields:
            np.ndarray: int16 mono audio chunk.
        """

        try:
            print(f"Starting TTS generation for text: {text[:50]}...")

            # Initialize Kokoro generator (NO split_pattern override)
            generator = self.pipeline(text, voice=voice, speed=speed)

            sample_rate = self.sample_rate
            frame = None

            # If user requests fixed-size streaming chunks
            if chunk_size_ms is not None:
                frame = int(sample_rate * (chunk_size_ms / 1000.0))

            total_chunks = 0

            # Iterate Kokoro chunks
            for i, (gs, ps, audio) in enumerate(generator):
                print(f"Generated audio chunk {i}, shape: {audio.shape}, dtype: {audio.dtype}")

                # Convert torch tensor to numpy
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()

                # Normalize float audio → int16
                if audio.dtype in (np.float32, np.float64):
                    audio = np.clip(audio, -1.0, 1.0)
                    audio = (audio * 32767).astype(np.int16)

                # ---- MODE 1: Real streaming (fixed-size chunks) ----
                if frame is not None:
                    start = 0
                    audio_len = len(audio)

                    while start < audio_len:
                        small_chunk = audio[start:start + frame]

                        if len(small_chunk) > 0:
                            total_chunks += 1
                            yield small_chunk

                        start += frame

                # ---- MODE 2: Native Kokoro chunking ----
                else:
                    total_chunks += 1
                    yield audio

            print(f"TTS generation completed. Total chunks: {total_chunks}")

        except Exception as e:
            print(f"Error in TTS generation: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


    
    def generate_full_audio(self, text, voice='af_heart', speed=1.0):
        """
        Generate complete audio (non-streaming)
        Returns concatenated audio array
        """
        audio_chunks = []
        for audio in self.stream_speech(text, voice, speed):
            audio_chunks.append(audio)
        
        if audio_chunks:
            return np.concatenate(audio_chunks)
        return np.array([], dtype=np.int16)
    
    def numpy_to_wav_bytes(self, audio_array):
        """Convert numpy array to WAV bytes"""
        byte_io = io.BytesIO()
        
        with wave.open(byte_io, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes = 16 bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_array.tobytes())
        
        byte_io.seek(0)
        return byte_io.read()