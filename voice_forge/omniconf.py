from pathlib import Path
from datetime import datetime
import os
import pytz
import logging
from dynaconf import Dynaconf


_NOW = datetime.now()
_BASE_DIR = Path(__file__).resolve().parent


def _get_start_ts(tz: str) -> datetime:
    return _NOW.astimezone(pytz.timezone(tz))


def _get_now_iso(tz: str) -> str:
    return datetime.now().astimezone(pytz.timezone(tz)).isoformat()


def _get_now_ts(tz: str) -> str:
    return datetime.now().astimezone(pytz.timezone(tz))


###################
# Create Settings #
###################
secrets_dir = os.environ.get("SECRETS_DIRECTORY") or ""

# Automatically detect .toml config files (excluding settings.toml)
settings_files = [
    path.as_posix() for path in _BASE_DIR.joinpath('settings_file').glob("*.toml") if path.stem != "settings"
]

config = Dynaconf(
    preload=[_BASE_DIR.joinpath("settings_file", "settings.toml").as_posix()],
    settings_files=settings_files,
    secrets=[] if not secrets_dir else list(Path(secrets_dir).glob("*.toml")),
    environments=True,
    envvar_prefix="VOICE_FORGE",
    load_dotenv=True,
    _get_now_ts=_get_now_ts,
    _get_now_iso=_get_now_iso,
    _get_start_ts=_get_start_ts,
    now=_NOW,
    partition_date=_NOW.strftime("%Y/%m/%d"),
    root_dir=_BASE_DIR.as_posix(),
    home_dir=Path.home().as_posix(),
    merge_enabled=True,
)

#########################
# Logger Initialization #
#########################
class DefaultFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt=fmt, datefmt=datefmt)

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, pytz.timezone(config.get("tz")))
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()

    def format(self, record):
        record.full_path = record.pathname
        return super().format(record)


logger = logging.getLogger(config.logger_name)
logger.setLevel(logging.INFO)

fmt = "[%(asctime)s] %(levelname)s [%(full_path)s]: %(message)s"
formatter = DefaultFormatter(fmt=fmt)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logger.level)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
