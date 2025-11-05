# src/config_env.py
import os
from dotenv import load_dotenv

def load_env():
    # Priority: explicit ENV var ENV_NAME -> choose a file
    env_name = os.getenv("ENV_NAME")  # e.g., "development", "production"
    candidates = [
        "config/.env.local",                      # personal overrides
        f"config/.env.{env_name}" if env_name else "",  # specific environment
        "config/.env",                            # shared default
    ]
    for p in candidates:
        if p and os.path.exists(p):
            load_dotenv(p, override=False)
