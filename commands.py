import subprocess
import sys

from dotenv import load_dotenv

COMMANDS: dict[str, tuple[str, str]] = {
    "generate_ai": (
        "ai_text_detector/data/ai_datasets.py",
        "Generate AI essays using prompt strategies",
    ),
    "download_datasets": (
        "ai_text_detector/data/human_datasets.py",
        "Parse combined_essays.jsonl and split into IvyPanda / ASAP2 / PERSUADE CSVs",
    ),
    "generate_embeddings": (
        "ai_text_detector/data/embeddings.py",
        "Generate embeddings of the datasets",
    ),
}


def print_help() -> None:
    print(__doc__)
    print("Available commands:\n")
    max_len = max(len(k) for k in COMMANDS)
    for name, (script, desc) in COMMANDS.items():
        print(f"  {name:<{max_len}}  →  {desc}")
    print()


def run_command(name: str, extra_args: list[str]) -> int:
    if name not in COMMANDS:
        print(f"Unknown command: '{name}'")
        print(f"Available: {', '.join(COMMANDS)}")
        return 1

    script, _ = COMMANDS[name]
    cmd = [sys.executable, script, *extra_args]
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    return result.returncode


def main() -> None:
    load_dotenv()

    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help", "help"):
        print_help()
        sys.exit(0)

    command, *extra = args
    sys.exit(run_command(command, extra))


if __name__ == "__main__":
    main()
