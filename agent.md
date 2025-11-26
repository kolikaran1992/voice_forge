# Project Overview

This project is an auto-initialized Python template managed by Poetry.
It provides a clean structure for configuration management using Dynaconf, along with support libraries like Jinja2 and pytz.

## 📁 Project Structure

```
project_root/
│
├── pyproject.toml               # Poetry configuration & dependencies
├── README.md                    # Project documentation
│
├── voice_forge/               # Main Python package
│   ├── __init__.py
│   ├── omniconf.py              # Base configuration loader using Dynaconf
│   └── settings_file/           # Dynaconf settings directory
│       └── settings.toml        # Default configuration
│
└── tests/                       # Unit tests directory
```

## ✅ What Each File Does

### `voice_forge/omniconf.py`
- Central config loader for the entire project
- Loads `settings.toml`
- Injects useful Jinja variables (`now`, timezone helpers)
- Sets base paths and timestamp values
- ✅ Initializes a global logger available across the project

To log messages:

```python
from voice_forge.omniconf import logger
logger.info("This is a log message")
```

### `voice_forge/settings_file/settings.toml`
- Contains default configuration values
- Uses Jinja2 templating inside Dynaconf
- Includes `logger_name` which is set to the project root name

Example:
```
[default]
now_iso = "@jinja {{this._get_now_iso(this.tz)}}"
start_ts = "@jinja {{this._get_start_ts(this.tz)}}"
tz = "Asia/Kolkata"
logger_name = "voice_forge"
base_data_path = "@jinja {{this.home_dir}}/Data/VOICE_FORGE"
```

If an AI agent needs to modify configuration behavior, it should edit:
- `voice_forge/omniconf.py` for logic or environment variable handling
- `voice_forge/settings_file/settings.toml` for changing configuration defaults

## 🔧 Extending the Project
- Add new settings in `voice_forge/settings_file/settings.toml`
- Add new Python modules inside `voice_forge/`
- Add tests inside `tests/`
