"""
Split existing GRIB files that are too large for eccodes/cfgrib's Windows
build to open (a large-file stat/seek bug that trips on files past ~2 GB --
confirmed on 1979_1980_1981-pressure.grib with cfgrib, three different
eccodes versions, and grib_to_netcdf, all failing the same way).

This works entirely at the raw byte level and never calls eccodes, so the
>2GB bug never comes up: Python's own file I/O has no such limit on 64-bit
Windows.

How it works: a GRIB file is a sequence of self-contained messages, each
starting with the 4-byte marker "GRIB" and carrying its own total length in
its header. This script scans the file message by message, groups whole
messages (never splitting one in half) into output files that stay under a
size limit, and writes each group out as its own smaller .grib file.

Usage:
    python scripts/split_large_grib.py <folder> [--max-gb 1.4]

Example:
    python scripts/split_large_grib.py "data/raw/era5/pressure" --max-gb 1.4
    python scripts/split_large_grib.py "data/raw/era5/single" --max-gb 1.4

Splits every *.grib file in <folder> larger than --max-gb, writing
<name>_part01.grib, <name>_part02.grib, ... next to the original. Files
already under the limit are left untouched.

IMPORTANT: originals are NOT deleted automatically. Once you've confirmed
the parts open correctly in 02_build_climatology.py, move or delete the
original (un-split) files out of the folder -- otherwise the pipeline will
glob both the original and its parts and double-count that data.
"""

import argparse
from pathlib import Path


def iter_messages(path):
    """Yield (start_offset, length) for each GRIB message in the file."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        pos = 0
        while pos < size:
            f.seek(pos)
            header = f.read(16)
            if len(header) < 16 or header[:4] != b"GRIB":
                raise ValueError(f"Expected 'GRIB' marker at offset {pos}, found {header[:4]!r}")
            edition = header[7]
            if edition == 1:
                length = int.from_bytes(header[4:7], "big")
            elif edition == 2:
                length = int.from_bytes(header[8:16], "big")
            else:
                raise ValueError(f"Unknown GRIB edition {edition} at offset {pos}")
            yield pos, length
            pos += length


def split_file(path, max_bytes):
    messages = list(iter_messages(path))
    print(f"{path.name}: {len(messages)} messages, {path.stat().st_size / 1e9:.2f} GB")

    part_num = 1
    out_path = path.with_name(f"{path.stem}_part{part_num:02d}{path.suffix}")
    out = open(out_path, "wb")
    written = 0

    with open(path, "rb") as src:
        for start, length in messages:
            if written + length > max_bytes and written > 0:
                out.close()
                print(f"  wrote {out_path.name} ({written / 1e9:.2f} GB)")
                part_num += 1
                out_path = path.with_name(f"{path.stem}_part{part_num:02d}{path.suffix}")
                out = open(out_path, "wb")
                written = 0
            src.seek(start)
            out.write(src.read(length))
            written += length

    out.close()
    print(f"  wrote {out_path.name} ({written / 1e9:.2f} GB)")
    print(f"  done: {part_num} parts\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--max-gb", type=float, default=1.4)
    args = parser.parse_args()

    max_bytes = int(args.max_gb * 1e9)
    files = sorted(p for p in args.folder.glob("*.grib") if "_part" not in p.stem)
    if not files:
        print(f"No .grib files found in {args.folder}")
        return

    for f in files:
        size = f.stat().st_size
        if size <= max_bytes:
            print(f"{f.name}: {size / 1e9:.2f} GB, under limit -- skipping")
            continue
        split_file(f, max_bytes)


if __name__ == "__main__":
    main()
