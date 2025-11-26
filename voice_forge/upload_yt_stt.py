import os
from huggingface_hub import HfApi, upload_folder
from voice_forge.omniconf import config, logger
import json

# ------------------------------
# CONFIGURATION
# ------------------------------
HF_TOKEN = config.hf.hf_token_write
REPO_ID = f"{config.hf.username}/{config.hf.ds_name.yt_tts}"
LOCAL_DATA_DIR = config.yt_tts_hf_base
# ------------------------------


def main():
    if HF_TOKEN is None:
        raise ValueError("No HF token found. Run: huggingface-cli login")

    logger.info(f"Uploading folder: {LOCAL_DATA_DIR}")
    logger.info(f"To repository: {REPO_ID}")

    # Uploads only changed files, keeps history
    upload_folder(
        repo_id=REPO_ID,
        folder_path=LOCAL_DATA_DIR,
        repo_type="dataset",
        token=HF_TOKEN,
    )

    logger.info("✅ Upload complete")


if __name__ == "__main__":
    main()
