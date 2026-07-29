from pathlib import Path
import sys

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is not installed.")
    print("Install it using: pip install pyyaml")
    sys.exit(1)


CONFIG_FILE = Path("configs/exp17/config_qwen35_9b_exp17.yaml")


REQUIRED_FIELDS = [
    ("experiment", "id"),
    ("experiment", "name"),
    ("model_path",),
    ("method",),
    ("training", "learning_rate"),
    ("training", "batch_size"),
    ("training", "gradient_accumulation_steps"),
    ("training", "num_epochs"),
    ("lora", "r"),
    ("lora", "alpha"),
    ("quantization", "type"),
]


def get_nested_value(data, keys):
    value = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def main():

    if not CONFIG_FILE.exists():
        print(f"ERROR: Config file not found: {CONFIG_FILE}")
        sys.exit(1)

    with CONFIG_FILE.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    missing = []

    for field in REQUIRED_FIELDS:
        value = get_nested_value(config, field)
        if value is None:
            missing.append(" -> ".join(field))

    if missing:
        print("\nConfiguration validation FAILED\n")
        for item in missing:
            print(f"Missing: {item}")
        sys.exit(1)

    print("\nConfiguration validation PASSED\n")

    print(f"Experiment : {config['experiment']['id']}")
    print(f"Model      : {config['model_path']}")
    print(f"Method     : {config['method']}")
    print(f"Epochs     : {config['training']['num_epochs']}")
    print(f"Batch Size : {config['training']['batch_size']}")
    print(f"LoRA Rank  : {config['lora']['r']}")


if __name__ == "__main__":
    main()
