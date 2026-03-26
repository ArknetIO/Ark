import sys

def main() -> int:
    n = 20_000_000
    if len(sys.argv) > 1:
        n = int(sys.argv[1])

    mod = 1_000_000_007
    acc = 0

    for i in range(n):
        acc = (acc + i) % mod

    print(f"py checksum: {acc}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())