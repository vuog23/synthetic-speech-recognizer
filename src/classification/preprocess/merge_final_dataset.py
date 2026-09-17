"""Build the Final dataset described by datasets/test.ipynb.

The output layout is::

    Final/
      train/{real,fake}/
      test/{real,fake}/
      val/{real,fake}/

Source split policy, matching the final cell in test.ipynb:

* ASVspoof and LibriSpeech real files retain their source train/test/val split.
* All VCTK files are real and are shuffled into 2/4 train, 1/4 test, 1/4 val.
* ASVspoof train/val fake files retain their source split. The combined source
  test fake pool (ASVspoof plus any LibriSpeech fake files) is shuffled 2/3
  into train and 1/3 into test. No files from that pool go to validation.

Files are hard-linked by default. This is fast and does not duplicate audio
bytes while keeping normal files under Final/. Use --mode copy if the source
and destination are on different volumes or independent copies are required.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal


SPLITS = ("train", "test", "val")
LABELS = ("real", "fake")
AUDIO_SUFFIXES = {".wav", ".flac"}
DATASETS_ROOT = Path(__file__).resolve().parents[3] / "datasets"


@dataclass(frozen=True)
class Sample:
    source: Path
    dataset: str
    source_split: str
    source_label: str
    final_split: str
    final_label: str


def audio_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing source directory: {directory}")
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )


def balanced_four_way(files: list[Path], randomizer: random.Random) -> dict[str, list[Path]]:
    """Return 2/4 train, 1/4 test and 1/4 val with all remainder assigned."""
    shuffled = files.copy()
    randomizer.shuffle(shuffled)
    # Train receives the rounded-up half; split the remainder between test and
    # validation. For a multiple of four this is exactly 2/4, 1/4, 1/4.
    train_size = (len(shuffled) + 1) // 2
    test_size = (len(shuffled) - train_size + 1) // 2
    sizes = {"train": train_size, "test": test_size, "val": len(shuffled) - train_size - test_size}
    start = 0
    result = {}
    for split in SPLITS:
        result[split] = shuffled[start:start + sizes[split]]
        start += sizes[split]
    assert start == len(shuffled)
    return result


def two_to_one(files: list[Path], randomizer: random.Random) -> dict[str, list[Path]]:
    """Return 2/3 train and 1/3 test, matching the notebook calculation."""
    shuffled = files.copy()
    randomizer.shuffle(shuffled)
    test_size = len(shuffled) // 3
    return {"train": shuffled[test_size:], "test": shuffled[:test_size]}


def source_samples(asv_root: Path, libri_root: Path, vctk_root: Path, seed: int) -> list[Sample]:
    randomizer = random.Random(seed)
    samples: list[Sample] = []
    fake_test_pool: list[tuple[Path, str]] = []

    for split in SPLITS:
        for label in LABELS:
            files = audio_files(asv_root / split / label)
            if label == "fake" and split == "test":
                fake_test_pool.extend((path, "asvspoof") for path in files)
            else:
                samples.extend(
                    Sample(path, "asvspoof", split, label, split, label)
                    for path in files
                )

    # LibriSpeech processed files are all bona-fide; support fake directories
    # too in case a future preprocessing stage adds them.
    for split in SPLITS:
        flat_files = audio_files(libri_root / split)
        samples.extend(
            Sample(path, "librispeech", split, "real", split, "real")
            for path in flat_files
        )
        fake_directory = libri_root / split / "fake"
        if fake_directory.is_dir():
            fake_test_pool.extend((path, "librispeech") for path in audio_files(fake_directory))

    for split, files in balanced_four_way(audio_files(vctk_root), randomizer).items():
        samples.extend(Sample(path, "vctk", "all", "real", split, "real") for path in files)

    fake_test_datasets = dict(fake_test_pool)
    for split, files in two_to_one(list(fake_test_datasets), randomizer).items():
        samples.extend(
            Sample(
                path,
                fake_test_datasets[path],
                "test",
                "fake",
                split,
                "fake",
            )
            for path in files
        )
    return samples


def expected_counts(samples: Iterable[Sample]) -> Counter[tuple[str, str]]:
    return Counter((sample.final_split, sample.final_label) for sample in samples)


def output_name(sample: Sample) -> str:
    # Dataset prefixes prevent any basename collision between source datasets.
    return f"{sample.dataset}__{sample.source.name}"


def link_or_copy(source: Path, destination: Path, mode: Literal["hardlink", "copy"]) -> None:
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError as error:
            raise OSError(
                f"Cannot hard-link {source} to {destination}. Use --mode copy "
                "when the paths are on different volumes."
            ) from error
    shutil.copy2(source, destination)


def build(samples: list[Sample], output: Path, mode: Literal["hardlink", "copy"]) -> None:
    for split in SPLITS:
        for label in LABELS:
            (output / split / label).mkdir(parents=True, exist_ok=False)

    names: set[Path] = set()
    manifest_rows = []
    for index, sample in enumerate(samples, start=1):
        destination = output / sample.final_split / sample.final_label / output_name(sample)
        if destination in names:
            raise ValueError(f"Duplicate output name: {destination.name}")
        names.add(destination)
        link_or_copy(sample.source, destination, mode)
        manifest_rows.append({
            "final_split": sample.final_split,
            "label": sample.final_label,
            "dataset": sample.dataset,
            "source_split": sample.source_split,
            "source_label": sample.source_label,
            "source": str(sample.source.resolve()),
            "output": str(destination.relative_to(output)),
        })
        if index % 5000 == 0 or index == len(samples):
            print(f"Created {index:,}/{len(samples):,} files", flush=True)

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)


def validate(output: Path, expected: Counter[tuple[str, str]]) -> None:
    actual = Counter()
    for split in SPLITS:
        for label in LABELS:
            count = len(audio_files(output / split / label))
            actual[(split, label)] = count
    if actual != expected:
        raise RuntimeError(f"Output counts do not match plan. Expected {expected}; got {actual}.")


def print_distribution(counts: Counter[tuple[str, str]]) -> None:
    print("\nFinal dataset distribution")
    print("split    real     fake     total")
    print("---------------------------------")
    for split in SPLITS:
        real, fake = counts[(split, "real")], counts[(split, "fake")]
        print(f"{split:<8} {real:>6,} {fake:>8,} {real + fake:>9,}")
    print("---------------------------------")
    print(f"total    {sum(counts[(split, 'real')] for split in SPLITS):>6,} "
          f"{sum(counts[(split, 'fake')] for split in SPLITS):>8,} "
          f"{sum(counts.values()):>9,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asvspoof", type=Path, default=DATASETS_ROOT / "ASVspoof_2019_LA" / "processed")
    parser.add_argument("--librispeech", type=Path, default=DATASETS_ROOT / "LibriSpeech_100" / "processed")
    parser.add_argument("--vctk", type=Path, default=DATASETS_ROOT / "VCTK")
    parser.add_argument("--output", type=Path, default=DATASETS_ROOT / "Final")
    parser.add_argument("--seed", type=int, default=42, help="Seed for VCTK and fake test-pool shuffles.")
    parser.add_argument("--mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--replace", action="store_true", help="Backup a non-empty output folder, then replace it.")
    parser.add_argument("--dry-run", action="store_true", help="Show the exact intended distribution without creating files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output == DATASETS_ROOT.resolve():
        raise ValueError("--output cannot be the datasets directory.")

    samples = source_samples(args.asvspoof.resolve(), args.librispeech.resolve(), args.vctk.resolve(), args.seed)
    counts = expected_counts(samples)
    print_distribution(counts)
    if args.dry_run:
        return 0

    if output.exists() and any(output.iterdir()):
        if not args.replace:
            raise FileExistsError(f"{output} is not empty. Re-run with --replace to keep a timestamped backup.")
        backup = output.with_name(f"{output.name}.backup-{datetime.now():%Y%m%d-%H%M%S}")
        print(f"Backing up existing output to {backup}")
        output.rename(backup)

    staging = output.with_name(f"{output.name}.building")
    if staging.exists():
        raise FileExistsError(f"Staging directory exists: {staging}. Inspect or remove it before running again.")
    try:
        build(samples, staging, args.mode)
        validate(staging, counts)
        # A file browser or another process can recreate an empty Final folder
        # after it was moved to its backup. Only remove an empty placeholder.
        if output.exists():
            try:
                output.rmdir()
            except OSError as error:
                raise RuntimeError(
                    f"Cannot promote {staging}: {output} was recreated and is not empty. "
                    "The completed staging dataset was left untouched."
                ) from error
        staging.rename(output)
    except BaseException:
        print(f"Build stopped. Partial files remain in {staging}", file=sys.stderr)
        raise

    print_distribution(counts)
    print(f"\nCreated: {output}")
    print(f"Manifest: {output / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
