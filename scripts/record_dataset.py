import sounddevice as sd
import soundfile as sf
import numpy as np
import os

DATASET_DIR = "data/audio_samples"
SAMPLERATE = 16000
CHANNELS = 1

os.makedirs(DATASET_DIR, exist_ok=True)

print("Audio Recording Started")

# ---------------- FIND LAST INDEX ----------------
existing_files = [
    f for f in os.listdir(DATASET_DIR)
    if f.startswith("recording_") and f.endswith(".wav")
]

if existing_files:
    indices = [int(f.split("_")[1].split(".")[0]) for f in existing_files]
    start_index = max(indices) + 1
else:
    start_index = 1

print(f"Starting from recording number: {start_index}")

# ---------------- USER INPUT ----------------
num_new = int(input("How many new recordings do you want to add? "))

# ---------------- RECORD LOOP ----------------
for i in range(start_index, start_index + num_new):

    input(f"\nPress ENTER to START recording sample {i}")
    print("Recording... Press ENTER to STOP.")

    frames = []

    def callback(indata, frames_count, time, status):
        if status:
            print(status)
        frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLERATE,
        channels=CHANNELS,
        callback=callback
    )

    stream.start()
    input()
    stream.stop()
    stream.close()

    if len(frames) == 0:
        print("No audio captured. Try again.")
        continue

    audio = np.concatenate(frames, axis=0)

    file_name = f"recording_{i}.wav"
    file_path = os.path.join(DATASET_DIR, file_name)

    sf.write(file_path, audio, SAMPLERATE)

    print("Saved:", file_path)

print("\nRecording session complete.")