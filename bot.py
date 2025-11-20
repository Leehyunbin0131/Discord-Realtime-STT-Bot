import discord
from discord.ext import commands
import discord.ext.voice_recv
from discord.ext.voice_recv import AudioSink, VoiceData
import os
import numpy as np
import multiprocessing
import queue
from stt_handler import run_stt_process
import datetime
from dotenv import load_dotenv


# Load environment variables
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")



intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

# Global Queues
audio_queue = None
result_queue = None
command_queue = None
stt_process = None

class STTSink(AudioSink):
    def __init__(self):
        super().__init__()
        print("STTSink initialized.")
    
    def wants_opus(self):
        return False

    def cleanup(self):
        print("STTSink cleanup.")
        pass

    def write(self, user, data: VoiceData):
        if audio_queue is None:
            return
            
        try:
            # Fast conversion using numpy
            audio_data = np.frombuffer(data.pcm, dtype=np.int16)
            audio_data = audio_data.reshape(-1, 2)
            mono_data = audio_data.mean(axis=1).astype(np.int16)
            resampled_data = mono_data[::3] # 48k -> 16k
            
            # Put into Multiprocessing Queue
            audio_queue.put((user.id, resampled_data.tobytes()))
            
        except Exception as e:
            print(f"Error in write: {e}")

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('------')
    
    # Start Result Listener
    bot.loop.create_task(process_results())

async def process_results():
    print("Result processing task started.")
    import json
    while True:
        try:
            # Non-blocking check
            if result_queue and not result_queue.empty():
                result = result_queue.get()
                print(json.dumps(result, ensure_ascii=False))
            else:
                await discord.utils.sleep_until(discord.utils.utcnow() + datetime.timedelta(milliseconds=10))
                # Or just await asyncio.sleep(0.01)
                import asyncio
                await asyncio.sleep(0.01)
        except Exception as e:
            print(f"Error in result loop: {e}")
            import asyncio
            await asyncio.sleep(1)

@bot.event
async def on_voice_state_update(member, before, after):
    # Check if user left the voice channel where the bot is
    if before.channel and not after.channel:
        # User left
        if command_queue:
            command_queue.put(("LEAVE", member.id))

@bot.command()
async def join(ctx):
    if ctx.author.voice:
        channel = ctx.author.voice.channel
        if ctx.voice_client is not None:
            return await ctx.voice_client.move_to(channel)
        
        vc = await channel.connect(cls=discord.ext.voice_recv.VoiceRecvClient)
        vc.listen(STTSink())
        await ctx.send(f"Joined {channel} and listening.")
    else:
        await ctx.send("You are not in a voice channel.")

@bot.command()
async def leave(ctx):
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        await ctx.send("Left the voice channel.")
    else:
        await ctx.send("I am not in a voice channel.")

if __name__ == "__main__":
    # Multiprocessing Support for Windows
    multiprocessing.freeze_support()
    
    # Initialize Queues
    audio_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    command_queue = multiprocessing.Queue()
    
    # Start STT Process
    stt_process = multiprocessing.Process(target=run_stt_process, args=(audio_queue, result_queue, command_queue))
    stt_process.daemon = True # Ensure it dies if main process dies
    stt_process.start()
    
    if not TOKEN:
        print("Error: DISCORD_TOKEN not found.")
    else:
        try:
            bot.run(TOKEN)
        except KeyboardInterrupt:
            pass
        finally:
            print("Terminating STT Process...")
            stt_process.terminate()
            stt_process.join()
