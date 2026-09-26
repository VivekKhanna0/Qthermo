"""``python -m qthermo`` -- reproduce published results from the command line.

    python -m qthermo              list the papers
    python -m qthermo fridge       reproduce one
    python -m qthermo all          reproduce all of them
"""

import sys
import time

from .papers import PAPERS, run


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print("Reproduce a published result:  python -m qthermo <name>   (or: all)\n")
        for name, (_, blurb) in PAPERS.items():
            print(f"  {name:<16} {blurb}")
        return 0
    names = list(PAPERS) if args == ["all"] else args
    for name in names:
        start = time.perf_counter()
        print(run(name).report())
        print(f"({time.perf_counter() - start:.1f} s)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
