from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multiplexing import generate_all_figures


def main() -> None:
    generate_all_figures(ROOT / "data" / "wave_case", ROOT / "figures")


if __name__ == "__main__":
    main()
