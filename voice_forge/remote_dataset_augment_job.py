import os
import json
from pathlib import Path
import shutil


from datasets import load_dataset
from huggingface_hub import (
    login,
    upload_folder,
    HfApi,
)

# -----------------------------------------------
# Optional dependency install (Kaggle-safe)
# -----------------------------------------------
try:
    import yt_dlp
    from pydub import AudioSegment
except ImportError:
    # Silence pip noise a bit on Kaggle
    os.system("pip install yt-dlp pydub > /dev/null")
    import yt_dlp
    from pydub import AudioSegment


OUTPUT_DIR = "/kaggle/working/output"


# -----------------------------------------------
# Config
# -----------------------------------------------
CREDS_PATH = "/kaggle/input/hf-token/hf_kaggle.json"
OUTPUT_DIR = "/kaggle/working/output"


def load_creds():
    with open(CREDS_PATH, "r") as f:
        return json.load(f)


def init_env(hf_token: str):
    login(token=hf_token)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------------------
# HF helpers — use split_audio/ presence as truth
# -----------------------------------------------
def load_processed_ids_from_repo(repo_id: str, token: str):
    """
    Derive processed video_ids from the presence of folders/files
    under split_audio/<video_id>/ in the HF dataset repo.
    """
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    processed_ids = set()
    for path in files:
        # e.g. "split_audio/abc123/utt_0001.wav"
        if path.startswith("split_audio/"):
            parts = path.split("/")
            if len(parts) >= 2:
                processed_ids.add(parts[1])

    return processed_ids


# -----------------------------------------------
# Audio processing
# -----------------------------------------------
def download_audio(video_id: str, wav_path: str):
    """Download YouTube audio as WAV if it doesn't exist yet."""
    wav_file = f"{wav_path}.wav"
    if os.path.exists(wav_file):
        print(f"✅ wav already exists: {wav_file}")
        return

    print(f"🔽 downloading audio for {video_id}...")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": wav_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "http_headers": {
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "en-US,en;q=0.5",
        },
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])


def split_audio(video_id, utterances, output_dir):
    """
    Download audio (if needed), split into utterances, and save to:

        output_dir/split_audio/<video_id>/<utt_id>.wav
    """
    wav_dir = os.path.join(output_dir, "wav")
    split_dir = os.path.join(output_dir, "split_audio", video_id)

    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)

    wav_path = os.path.join(wav_dir, video_id)

    # 1) ensure audio downloaded
    download_audio(video_id, wav_path)

    # 2) load full audio
    print("🔊 loading audio...")
    audio = AudioSegment.from_wav(f"{wav_path}.wav")

    # 3) split + export
    print("✂️ splitting into utterances...")
    saved = 0

    for utt in utterances:
        utt_id = utt["utt_id"]
        start = utt["start_ms"]
        end = utt["end_ms"]
        out_path = os.path.join(split_dir, f"{utt_id}.wav")

        if os.path.exists(out_path):
            print(f"✅ already exists: {out_path}")
            saved += 1
            continue

        segment = audio[start:end]
        segment.export(out_path, format="wav")
        saved += 1

    print(f"✅ saved {saved} chunks for {video_id}")
    return saved


# -----------------------------------------------
# Data / pipeline helpers
# -----------------------------------------------
def select_next_video(repo_id: str, token: str):
    """Pick a video_id that doesn't have split_audio/<id> on HF yet."""
    metadata_ds = load_dataset(
        repo_id,
        data_files="metadata.jsonl",
        split="train",
        streaming=True,
        token=token,
    )
    all_video_ids = {row["id"] for row in metadata_ds}

    processed_ids = load_processed_ids_from_repo(repo_id, token)
    unprocessed = all_video_ids - processed_ids

    if not unprocessed:
        raise RuntimeError(
            "✅ No unprocessed videos left (all have split_audio/* on HF)"
        )

    return next(iter(unprocessed))


def load_utterances_for_video(repo_id: str, video_id: str, token: str):
    utterances_ds = load_dataset(
        repo_id,
        data_files=f"utterances/{video_id}/utterances.jsonl",
        split="train",
        streaming=True,
        token=token,
    )
    return list(utterances_ds)


def upload_split_audio(repo_id: str, token_write: str, video_id: str):
    upload_folder(
        repo_id=repo_id,
        folder_path=f"{OUTPUT_DIR}/split_audio/{video_id}",
        path_in_repo=f"split_audio/{video_id}",
        repo_type="dataset",
        token=token_write,
    )


def cleanup_output(output_dir=OUTPUT_DIR):
    """
    Safely remove all contents of the Kaggle working output directory.

    - deletes files and folders inside output_dir
    - preserves the directory itself
    - ignores errors (common on Kaggle)
    """
    if not os.path.exists(output_dir):
        print(f"⚠️ output dir does not exist: {output_dir}")
        return

    print(f"🧹 cleaning output: {output_dir}")

    # remove everything inside output_dir
    for item in os.listdir(output_dir):
        path = os.path.join(output_dir, item)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            else:
                shutil.rmtree(path, ignore_errors=True)
        except Exception as e:
            print(f"⚠️ could not remove {path}: {e}")

    print("✅ cleanup complete")


# -----------------------------------------------
# Main
# -----------------------------------------------
def main():
    creds = load_creds()

    HF_TOKEN = creds["hf_token"]
    HF_TOKEN_WRITE = creds["hf_token_write"]
    REPO_ID = creds["repo_id"]

    init_env(HF_TOKEN)

    # 1) select a video that does NOT have split_audio/<id> on HF
    video_id = select_next_video(REPO_ID, HF_TOKEN)
    print("🔧 processing:", video_id)

    # 2) load utterances for that video
    utterances = load_utterances_for_video(REPO_ID, video_id, HF_TOKEN)

    # 3) download + split locally
    count = split_audio(video_id, utterances, OUTPUT_DIR)
    print("✅ chunks saved:", count)

    # 4) upload split audio to HF
    upload_split_audio(REPO_ID, HF_TOKEN_WRITE, video_id)

    print("✅ HF updated for", video_id)
    # 5) cleanup output
    cleanup_output()
    print(f"{OUTPUT_DIR} cleaned")


if __name__ == "__main__":
    main()
