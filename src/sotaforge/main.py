"""Main for SOTAforge tool."""

import asyncio
import sys

from sotaforge.client import call_produce_sota


def main() -> int:
    """Generate a SOTA on a given topic."""
    if len(sys.argv) < 2:
        print("Usage: sota <topic>")
        print("Example: sota 'Edge Computing'")
        return 1

    topic = " ".join(sys.argv[1:])
    print(f"🚀 Generating SOTA on: {topic}\n")
    result = asyncio.run(call_produce_sota(topic))
    print("\n✅ Done!")
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
