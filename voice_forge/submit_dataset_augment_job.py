import uuid
import random
import time
from pathlib import Path
from crongle import KernelLauncher, KAGGLE_USER_NAME
from voice_forge.omniconf import config, logger

PRIVATE_DATASET = f"{KAGGLE_USER_NAME}/hf-token"

SCRIPT_PATH = "voice_forge/remote_dataset_augment_job.py"  # code to run remotely
OUTPUT_BASE_DIR = config.yt_tts_data_augment_output  # results downloaded here


def wait_with_jitter(min_minutes=2, max_minutes=15):
    jitter_seconds = random.randint(min_minutes * 60, max_minutes * 60)
    logger.info(
        f"Waiting {jitter_seconds // 60}m {jitter_seconds % 60}s before submission..."
    )
    time.sleep(jitter_seconds)


def submit():
    run_id = str(uuid.uuid4())[:8]
    kernel_name = f"crongle-job-{run_id}"

    output_dir = Path(OUTPUT_BASE_DIR).joinpath(kernel_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"🚀 Submitting {kernel_name} ...")

    job_id = KernelLauncher().submit_job(
        kernel_name=kernel_name,
        script_path=SCRIPT_PATH,
        output_folder=str(output_dir),
        timeout=3600,
        interval_amount=5,
        interval_unit="minute",
        kernel_kwargs={
            "title": kernel_name,
            "is_private": True,
            "enable_gpu": False,
            "enable_internet": True,
            "dataset_sources": [PRIVATE_DATASET],  # REQUIRED for private datasets
        },
        slack_channel_id=config.crongle.slack.channel_id,  # Optional: Slack channel ID for notifications
        slack_bot_token=config.slack.bot_token,  # Optional: Slack bot token for notifications
    )

    logger.info(f"✅ Submitted: {kernel_name}")
    logger.info(f"🔗 https://www.kaggle.com/code/{KAGGLE_USER_NAME}/{kernel_name}")
    logger.info(f"📁 Output will be in: {output_dir}")

    return job_id


if __name__ == "__main__":
    wait_with_jitter()
    submit()
