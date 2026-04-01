import json
from jiwer import wer, process_words, visualize_alignment
import re
from num2words import num2words
import os
from collections import Counter
from nltk.stem import WordNetLemmatizer
import nltk
from difflib import SequenceMatcher

# download once if not already installed
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()

# counters for error analysis
confusion_counter = Counter()
error_word_counter = Counter()

# counter for corrected WER
corrected_total_wer = 0

# function to compute similarity ratio for potential auto-corrections
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# NORMALIZATION 
def normalize_text(text):
    text = text.lower()

    def replace_number(match):
        return num2words(int(match.group()))

    text = re.sub(r'\b\d+\b', replace_number, text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# LEMMATIZATION 
LEMMA_BLOCKLIST = {"was", "has", "is", "his", "its", "as"}

def lemmatize_text(text):
    words = text.split()
    lemmas = []
    for word in words:
        if word in LEMMA_BLOCKLIST:
            lemmas.append(word)   # skip lemmatization, keep as-is
        else:
            lemmas.append(lemmatizer.lemmatize(word))
    return " ".join(lemmas)


# APPLY AUTO CORRECTIONS 
def apply_corrections(text, corrections):
    words = text.split()
    corrected = [corrections.get(word, word) for word in words]
    return " ".join(corrected)


# LOAD DATA
with open("outputs/predictions/all_predictions.json") as f:
    predictions = json.load(f)

with open("data/dataset_manifest.json") as f:
    dataset = json.load(f)

references = {}

for item in dataset:
    file_path = item["audio_filepath"]
    file_name = os.path.basename(file_path)
    references[file_name] = item["text"]


# LOAD AUTO CORRECTIONS (if exists from previous run)
auto_corrections = {}
if os.path.exists("outputs/evaluations/auto_corrections.json"):
    with open("outputs/evaluations/auto_corrections.json") as f:
        auto_corrections = json.load(f)
    print(f"Loaded {len(auto_corrections)} auto corrections from previous run.\n")
else:
    print("No auto_corrections.json found — corrected WER will match raw WER on this run.\n")


total_wer = 0
count = 0
evaluation_results = {}

alignment_dir = "outputs/alignments"
os.makedirs(alignment_dir, exist_ok=True)

# MAIN EVALUATION 
for item in predictions:

    file_name = item["file_name"]
    predicted = item["transcript"]

    if file_name in references:

        reference = references[file_name]

        # normalization
        reference = normalize_text(reference)
        predicted = normalize_text(predicted)

        # lemmatization
        reference = lemmatize_text(reference)
        predicted = lemmatize_text(predicted)

        error = wer(reference, predicted)
        result = process_words(reference, predicted)

        # APPLY AUTO CORRECTIONS 
        corrected_predicted = apply_corrections(predicted, auto_corrections)
        corrected_error = wer(reference, corrected_predicted)

        print(f"{file_name} | WER: {error:.2f} | Corrected WER: {corrected_error:.2f}")

        ref_words = reference.split()
        pred_words = predicted.split()

        # ERROR ANALYSIS 
        for alignment in result.alignments:
            for chunk in alignment:

                if chunk.type == "substitute":

                    ref_segment = ref_words[chunk.ref_start_idx:chunk.ref_end_idx]
                    pred_segment = pred_words[chunk.hyp_start_idx:chunk.hyp_end_idx]

                    for r, p in zip(ref_segment, pred_segment):
                        confusion_counter[(r, p)] += 1
                        error_word_counter[r] += 1


        # SAVE ALIGNMENT
        corrected_result = process_words(reference, corrected_predicted)
        alignment_text = visualize_alignment(corrected_result)
       

        alignment_file = os.path.join(
            alignment_dir,
            file_name.replace(".wav", "_alignment.txt")
        )

        with open(alignment_file, "w") as f:
            f.write(alignment_text)
        
        # APPLY CORRECTIONS TO SEGMENTS 
        corrected_segments = []
        for seg in item.get("segments", []):
            seg_text = normalize_text(seg["text"])
            seg_text = lemmatize_text(seg_text)
            corrected_seg_text = apply_corrections(seg_text, auto_corrections)
            corrected_segments.append({
                **seg,
                "corrected_text": corrected_seg_text
            })
    


        #  STORE FILE RESULTS 
        evaluation_results[file_name] = {
            "wer": round(error, 2),
            "corrected_wer": round(corrected_error, 2),
            "reference": reference,
            "prediction": predicted,
            "corrected_prediction": corrected_predicted,
            "segments": corrected_segments,  
            "substitutions": result.substitutions,
            "deletions": result.deletions,
            "insertions": result.insertions,
            "alignment_file": alignment_file
        }

        total_wer += error
        corrected_total_wer += corrected_error
        count += 1


# FINAL DATASET METRICS 
if count > 0:

    avg_wer = total_wer / count
    avg_corrected_wer = corrected_total_wer / count

    print("\nAverage Dataset WER:           ", round(avg_wer, 2))
    print("Average Corrected Dataset WER: ", round(avg_corrected_wer, 2))

    evaluation_results["average_wer"] = round(avg_wer, 2)
    evaluation_results["average_corrected_wer"] = round(avg_corrected_wer, 2)

    with open("outputs/evaluations/evaluation_results.json", "w") as f:
        json.dump(evaluation_results, f, indent=4)


# CONFUSION MATRIX 
confusion_dict = {
    f"{ref}->{pred}": freq
    for (ref, pred), freq in confusion_counter.items()
}

with open("outputs/evaluations/confusion_matrix.json", "w") as f:
    json.dump(confusion_dict, f, indent=4)


# MOST ERROR-PRONE WORDS 
top_problem_words = dict(error_word_counter.most_common(20))

with open("outputs/evaluations/top_problem_words.json", "w") as f:
    json.dump(top_problem_words, f, indent=4)


print("\nTop ASR Confusions:")
for (ref, pred), freq in confusion_counter.most_common(10):
    print(f"{ref} -> {pred} : {freq}")

print("\nMost Problematic Words:")
for word, freq in error_word_counter.most_common(10):
    print(f"{word} : {freq}")

# AUTO CORRECTION 

THRESHOLD = 1
SIM_THRESHOLD = 0.6   # key filter
HIGH_FREQ_THRESHOLD = 3      # if confusion happens this many times, skip similarity check
HIGH_FREQ_SIM_THRESHOLD = 0.4  # still need some minimum similarity even for high freq


STOPWORDS = {
    "the","and","a","an","is","are","was","were","it","if","of","to","in","on","for","with","but"
}
CONTEXT_SENSITIVE_BLOCKLIST = {
    "test",     # "blood test" is correct — don't replace with "rest"
    "rest",     # valid standalone in clinical context
    "check",    # "check hemoglobin" is correct
    "water",    # "drink water" is correct, "saltwater" is context-specific
}

auto_corrections = {}
blocklist_skipped = []


for (ref, pred), freq in confusion_counter.items():

    if freq < THRESHOLD:
        continue

    if ref in STOPWORDS or pred in STOPWORDS:
        continue

    # block medically ambiguous words
    if ref in CONTEXT_SENSITIVE_BLOCKLIST or pred in CONTEXT_SENSITIVE_BLOCKLIST:
        blocklist_skipped.append(f"{pred} -> {ref} (freq={freq})")
        continue

    # high frequency confusions — relax similarity requirement
    if freq >= HIGH_FREQ_THRESHOLD:
        if similarity(ref, pred) >= HIGH_FREQ_SIM_THRESHOLD:
            auto_corrections[pred] = ref
            continue


    # similarity filter (MAIN FIX)
    if similarity(ref, pred) < SIM_THRESHOLD:
        continue

    auto_corrections[pred] = ref


with open("outputs/evaluations/auto_corrections.json", "w") as f:
    json.dump(auto_corrections, f, indent=4)

print("\nFiltered Auto Corrections:")
print(auto_corrections)

if blocklist_skipped:
    print("\nSkipped (context-sensitive blocklist):")
    for entry in blocklist_skipped:
        print(f"  {entry}")
