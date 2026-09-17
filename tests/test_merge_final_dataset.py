import csv
import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "src" / "classification" / "preprocess" / "merge_final_dataset.py"
SPEC = importlib.util.spec_from_file_location("merge_final_dataset", SCRIPT)
merge = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = merge
SPEC.loader.exec_module(merge)


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(path.name.encode())


def test_split_plan_matches_notebook_policy_and_builds_hardlinks(tmp_path: Path):
    asv = tmp_path / "asv"
    libri = tmp_path / "libri"
    vctk = tmp_path / "vctk"
    for split, real, fake in (("train", 2, 3), ("test", 3, 6), ("val", 4, 5)):
        for index in range(real):
            touch(asv / split / "real" / f"asv-real-{split}-{index}.wav")
        for index in range(fake):
            touch(asv / split / "fake" / f"asv-fake-{split}-{index}.wav")
    for split, count in (("train", 7), ("test", 8), ("val", 9)):
        for index in range(count):
            touch(libri / split / f"libri-{split}-{index}.wav")
    for index in range(12):
        touch(vctk / f"{index:07d}.flac")

    samples = merge.source_samples(asv, libri, vctk, seed=7)
    counts = merge.expected_counts(samples)
    # 2:1:1 VCTK allocation and the notebook's test-fake redistribution.
    assert counts == {
        ("train", "real"): 2 + 7 + 6,
        ("train", "fake"): 3 + 4,
        ("test", "real"): 3 + 8 + 3,
        ("test", "fake"): 2,
        ("val", "real"): 4 + 9 + 3,
        ("val", "fake"): 5,
    }

    output = tmp_path / "Final.building"
    merge.build(samples, output, "hardlink")
    merge.validate(output, counts)
    with (output / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == sum(counts.values())
    assert {row["dataset"] for row in rows} == {"asvspoof", "librispeech", "vctk"}
    asv_output = output / "train" / "real" / "asvspoof__asv-real-train-0.wav"
    asv_source = asv / "train" / "real" / "asv-real-train-0.wav"
    assert asv_output.stat().st_ino == asv_source.stat().st_ino


def test_non_multiple_splits_assign_every_file(tmp_path: Path):
    files = []
    for index in range(7):
        path = tmp_path / f"{index}.flac"
        touch(path)
        files.append(path)
    portions = merge.balanced_four_way(files, merge.random.Random(42))
    assert {split: len(items) for split, items in portions.items()} == {"train": 4, "test": 2, "val": 1}
    assert len({item for items in portions.values() for item in items}) == 7
