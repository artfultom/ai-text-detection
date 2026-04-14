def log(msg: str, level: int = 0):
    indent = "  " * level
    print(f"{indent}{msg}")


def section(title: str):
    print(f"\n{'=' * 10} {title} {'=' * 10}")
