from pathlib import Path

def load_greetings(path: Path) -> set[str]:
    with open(path, encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}

_BASE = Path(__file__).parent

GREETINGS_SET = load_greetings(_BASE / "greetings.txt")
