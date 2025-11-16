"""
Legacy format detection and migration instructions.

Detects TACO v1 formats (.taco, .tortilla, tacofoundation:) and provides
clear migration path to v2 (.tacozip).
"""


def is_legacy_format(path: str) -> bool:
    """Check if path is legacy TACO v1 format."""
    return path.endswith((".taco", ".tortilla")) or path.startswith("tacofoundation:")


def raise_legacy_error(path: str) -> None:
    """
    Raise error with migration instructions for legacy format.

    Always raises ValueError with colored terminal output showing:
    - Why it failed (v1 not supported in v2+)
    - How to read legacy files (install tacoreader<1.0)
    - How to migrate (use tacotoolbox 2.0)
    """
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    message = (
        f"{YELLOW}⚠️  {BOLD}Legacy TACO v1 format detected:{RESET} {CYAN}{path}{RESET}\n\n"
        f"{RED}tacoreader 2.0+ does not support v1 formats "
        f"(.taco, .tortilla, tacofoundation:).{RESET}\n\n"
        f"{BOLD}To read legacy files:{RESET}\n"
        f"  1. Install tacoreader v0.x:\n"
        f"       {GREEN}pip install 'tacoreader<1.0'{RESET}\n\n"
        f"  2. Migrate your dataset to {CYAN}.tacozip{RESET} format using "
        f"{CYAN}tacotoolbox 2.0{RESET}\n\n"
        f"  3. {BOLD}{YELLOW}IMPORTANT:{RESET} Use the legacy import path:\n\n"
        f"         {GREEN}import tacoreader.v1 as tacoreader{RESET}\n\n"
        f"     instead of:\n"
        f"         {RED}import tacoreader{RESET}\n\n"
        f"{BOLD}We recommend migrating to the new PIT-based format "
        f"for better performance.{RESET}"
    )

    raise ValueError(message)
