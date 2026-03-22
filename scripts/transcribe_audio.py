import whisper 
import torch
import json
import subprocess
import os


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)
# Normalize the audio file to ensure it is in the correct format for Whisper
USE_NORMALIZATION = False
def normalize_audio(input_file, output_file):
    command = [
        "ffmpeg",
        "-i", input_file,
        "-ar", "16000",
        "-ac", "1",
        "-y",
        output_file
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

model = whisper.load_model("small").to(device)

audio_folder = "data/audio_samples"
output_folder = "outputs/predictions/individual"

# create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

 
all_results = []
# loop through all audio files
for file in os.listdir(audio_folder):

    if file.endswith(".wav"):

        audio_path = os.path.join(audio_folder, file)

        if USE_NORMALIZATION:
          normalized_path = os.path.join(audio_folder, f"{file}_norm.wav")
          normalize_audio(audio_path, normalized_path)
          audio_for_model = normalized_path
        else:
          audio_for_model = audio_path


        print(f"\nProcessing: {file}")
        
        result = model.transcribe(audio_for_model, language="en", beam_size=5, initial_prompt="Medical conversation: doctor and patient discussing symptoms and treatment. Keywords: fever, cough, headache, nausea, dizziness, fatigue, chest pain, stomach pain, blood pressure, diabetes, hypertension, paracetamol, ibuprofen, amoxicillin, antibiotics, tablets, prescription, dosage.")
        # print("\nFull Transcript:")
        # print(result["text"])
        print(f"\nTranscribed: {file}")


        clean_segments = []

        for i, seg in enumerate(result["segments"]):
            clean_segments.append({
                "segment_id": i + 1,
                "speaker": "unknown",
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"]
            })

        # print("\nSegments:")

        # for seg in clean_segments:
        #     print(f"Segment {seg['segment_id']}: {seg['text']}")

        output = {
            "file_name": file,
            "transcript": result["text"],
            "segments": clean_segments
        }
        
        # Stores all outputs in one file 
        all_results.append(output)

        output_file = os.path.join(output_folder, file.replace(".wav", ".json"))
        
        # create json output for each file
        with open(output_file, "w") as f:
            json.dump(output, f, indent=4)

        print(f"\nTranscript saved to {output_file}")

        # create json output for all files
        with open("outputs/predictions/all_predictions.json", "w") as f:
            json.dump(all_results, f, indent=4)
