import os
import toml

# Path to the config.toml file
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.toml")

try:
    with open(CONFIG_PATH, "r") as f:
        config = toml.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Configuration file not found at: {CONFIG_PATH}")

# Function to load environment variables, overriding TOML values if set
def load_env_config(config):
    """Loads environment variables, overriding TOML values if they exist."""
    def get_env_or_toml(section, key, default=None):
        """Gets the value from environment variable or TOML, with an optional default."""
        env_var_name = f"{section.upper()}_{key.upper()}" # Construct env var name
        return os.environ.get(env_var_name) or config.get(section, {}).get(key, default)


    # API Keys and Credentials
    config["api"] = config.get("api", {}) # Ensure the 'api' section exists
    config["api"]["mistral_api_key"] = get_env_or_toml("api", "mistral_api_key")
    config["api"]["huggingface_api_key"] = get_env_or_toml("api", "huggingface_api_key")
    config["api"]["google_cloud_credentials"] = get_env_or_toml("api", "google_cloud_credentials")
    config["api"]["nebius_api_key"] = get_env_or_toml("api", "nebius_api_key")

    # Example of overriding other settings (e.g., model parameters)
    config["model"] = config.get("model", {})
    config["model"]["embedding_model_name"] = get_env_or_toml("model", "embedding_model_name")
    config["model"]["vector_database_type"] = get_env_or_toml("model", "vector_database_type")


# Load environment variables, potentially overriding TOML values
load_env_config(config)


# Access configuration parameters (after loading env variables)
PROJECT_NAME = config["general"]["project_name"]
DEBUG_MODE = config["general"]["debug_mode"]

# Data paths (construct absolute paths)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", config["data"]["data_dir"])
MEDICAL_KNOWLEDGE_BASE_PATH = os.path.join(DATA_DIR, config["data"]["medical_knowledge_base_path"])
# ... other data paths ...

# Model parameters
EMBEDDING_MODEL_NAME = config["model"]["embedding_model_name"]
VECTOR_DATABASE_TYPE = config["model"]["vector_database_type"]
# ... other model parameters ...

# API Keys
MISTRAL_API_KEY = config["api"]["mistral_api_key"]
# ... other API keys ...

# ... other config parameters ...



# Validation function
def validate_config():
    """Validates essential configuration parameters."""
    essential_paths = []  # Example
    for path in essential_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Essential file not found: {path}. Check config.toml and data paths.")

    essential_api_keys = [MISTRAL_API_KEY] # Example
    for key, value in config["api"].items():
        if key in essential_api_keys and value is None:
            raise ValueError(f"Essential API key '{key}' is not set. Set the '{key.upper()}' environment variable.")

# Validate configuration on import
validate_config()