import subprocess
import sys
import json
import os


def run_script(script_path, label):
    print(f"[1/3] {label}..." if "transcribe" in script_path else f"[2/3] {label}...")

    result = subprocess.run(
        [sys.executable, script_path],
        stdout=subprocess.DEVNULL,   # suppress all print output
        stderr=None
    )

    if result.returncode != 0:
        print(f"ERROR: {script_path} failed. Pipeline stopped.")
        sys.exit(1)

    print(f"      Done.")


def save_pipeline_output():
    print(f"[3/3] Saving pipeline output...")

    results_path = "outputs/evaluations/evaluation_results.json"
    predictions_path = "outputs/predictions/all_predictions.json"
    output_dir = "outputs/pipeline"
    os.makedirs(output_dir, exist_ok=True)

    with open(results_path) as f:
        results = json.load(f)

    with open(predictions_path) as f:
        predictions = json.load(f)

    pred_index = {item["file_name"]: item for item in predictions}

    pipeline_output = []

    for file_name, data in results.items():

        if file_name in ("average_wer", "average_corrected_wer"):
            continue

        raw_segments = pred_index.get(file_name, {}).get("segments", [])
        corrected_segments = data.get("segments", [])   # from evaluation_results

        merged_segments = []
        for i, seg in enumerate(raw_segments):
            corrected_text = corrected_segments[i].get("corrected_text", "") if i < len(corrected_segments) else ""
            merged_segments.append({
                "segment_id": seg["segment_id"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "corrected_text": corrected_text
            })

        pipeline_output.append({
            "file_name": file_name,
            "raw_transcript": pred_index.get(file_name, {}).get("transcript", ""),
            "corrected_transcript": data.get("corrected_prediction", ""),
            "segments": merged_segments,
            "wer": data["wer"],
            "corrected_wer": data["corrected_wer"]
        })

    with open(os.path.join(output_dir, "pipeline_output.json"), "w") as f:
        json.dump(pipeline_output, f, indent=4)

    print(f"      Done.\n")


def print_summary():
    results_path = "outputs/evaluations/evaluation_results.json"

    with open(results_path) as f:
        results = json.load(f)

    avg_wer = results.get("average_wer", "N/A")
    avg_corrected = results.get("average_corrected_wer", "N/A")
    total = len(results) - 2

    print(f"{'='*47}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*47}")
    print(f"  Files processed    : {total}")
    print(f"  Average WER        : {avg_wer}")
    print(f"  Average Corrected  : {avg_corrected}")
    print(f"  Output saved to    : outputs/pipeline/pipeline_output.json")
    print(f"{'='*47}\n")

    print(f"  {'File':<25} {'WER':>6} {'Corrected':>10}")
    print(f"  {'-'*43}")
    for file_name, data in results.items():
        if file_name in ("average_wer", "average_corrected_wer"):
            continue
        print(f"  {file_name:<25} {data['wer']:>6.2f} {data['corrected_wer']:>10.2f}")


# ---------------- PIPELINE ----------------
if __name__ == "__main__":

    run_script("scripts/transcribe_audio.py", "Transcribing audio files")
    run_script("scripts/evaluate_wer.py",     "Evaluating and applying corrections")
    save_pipeline_output()
    print_summary()