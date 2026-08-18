#!/usr/bin/env python3
"""Build, validate, and run the repeated Levenshtein benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import shutil
import struct
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ADAPTERS = ("stringzilla", "rapidfuzz", "symspell", "fst", "tantivy", "lucene")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dictionary", type=Path)
    parser.add_argument("--stringzilla-root", type=Path, required=True)
    parser.add_argument("--rapidfuzz-root", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=Path("target/levenshtein"))
    parser.add_argument("--query-count", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=243)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--stringzilla-modes", default="cold,warm,steady,latency")
    args = parser.parse_args()
    if args.query_count < 1 or args.repeats < 1:
        parser.error("query count and repetitions must be positive")
    for name in ("dictionary", "stringzilla_root", "rapidfuzz_root"):
        setattr(args, name, getattr(args, name).resolve())
    args.build_dir = args.build_dir.resolve()
    return args


def run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env={**os.environ, **(env or {})},
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout}\n{completed.stderr}"
        )
    return completed


def compile_products(args: argparse.Namespace, root: Path) -> dict[str, Path | str]:
    build = args.build_dir
    binary = build / "bin"
    binary.mkdir(parents=True, exist_ok=True)
    products: dict[str, Path | str] = {
        "queries": binary / "queries",
        "stringzilla": binary / "stringzilla",
        "rapidfuzz": binary / "rapidfuzz",
        "rust": binary / "bench_levenshtein",
        "lucene_classpath": build / "lucene-classpath.txt",
    }
    if args.skip_build:
        missing = [str(path) for path in products.values() if isinstance(path, Path) and not path.exists()]
        if missing:
            raise RuntimeError("missing build products: " + ", ".join(missing))
        products["lucene_classpath"] = Path(products["lucene_classpath"]).read_text().strip()
        return products

    stringzilla_build = build / "stringzilla"
    run(
        ["cmake", "-S", str(args.stringzilla_root), "-B", str(stringzilla_build), "-DCMAKE_BUILD_TYPE=Release"],
        cwd=root,
    )
    run(["cmake", "--build", str(stringzilla_build), "-j", "--target", "stringzillas_cpus_static"], cwd=root)
    stringzilla_library = next(stringzilla_build.rglob("libstringzillas_cpus_static.a"), None)
    forkunion_library = next(stringzilla_build.rglob("libforkunion_static.a"), None)
    if not stringzilla_library or not forkunion_library:
        raise RuntimeError("StringZilla build did not produce the expected static libraries")

    cxx_flags = ["g++", "-std=c++20", "-O3", "-DNDEBUG", "-march=native"]
    include = args.stringzilla_root / "include"
    run(
        [*cxx_flags, "-I", str(include), str(root / "levenshtein/queries.cpp"), "-o", str(products["queries"])],
        cwd=root,
    )
    run(
        [
            *cxx_flags,
            "-DSZ_DYNAMIC_DISPATCH=1",
            "-I",
            str(include),
            "-I",
            str(args.stringzilla_root / "forkunion/include"),
            str(root / "levenshtein/stringzilla.cpp"),
            str(stringzilla_library),
            str(forkunion_library),
            "-pthread",
            "-o",
            str(products["stringzilla"]),
        ],
        cwd=root,
    )
    run(
        [
            *cxx_flags,
            "-I",
            str(args.rapidfuzz_root),
            str(root / "levenshtein/rapidfuzz.cpp"),
            "-o",
            str(products["rapidfuzz"]),
        ],
        cwd=root,
    )

    cargo_target = build / "cargo"
    cargo = run(
        [
            "cargo",
            "build",
            "--release",
            "--locked",
            "--features",
            "bench_levenshtein",
            "--bench",
            "bench_levenshtein",
            "--message-format=json",
        ],
        cwd=root,
        env={"CARGO_TARGET_DIR": str(cargo_target), "RUSTFLAGS": "-C target-cpu=native"},
    )
    executable = None
    for line in cargo.stdout.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if message.get("target", {}).get("name") == "bench_levenshtein" and message.get("executable"):
            executable = Path(message["executable"])
    if not executable:
        raise RuntimeError("Cargo did not report the Levenshtein benchmark executable")
    shutil.copy2(executable, products["rust"])

    lucene = build / "lucene"
    shutil.copytree(root / "levenshtein/lucene", lucene, dirs_exist_ok=True)
    dependencies = build / "lucene-dependencies"
    run(
        [
            "mvn",
            "-q",
            "-f",
            str(lucene / "pom.xml"),
            "package",
            "dependency:copy-dependencies",
            f"-DoutputDirectory={dependencies}",
        ],
        cwd=root,
    )
    jars = sorted(dependencies.glob("*.jar"))
    classes = lucene / "target/classes"
    if not jars or not classes.is_dir():
        raise RuntimeError("Maven did not produce Lucene classes and dependencies")
    classpath = os.pathsep.join([str(classes), *(str(jar) for jar in jars)])
    Path(products["lucene_classpath"]).write_text(classpath)
    products["lucene_classpath"] = classpath
    return products


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    completed = run(["git", "-C", str(path), "rev-parse", "HEAD"], cwd=path)
    return completed.stdout.strip() or None


def version(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
    except OSError:
        return None
    output = completed.stdout.strip() or completed.stderr.strip()
    return output.splitlines()[0] if output else None


def machine() -> dict[str, object]:
    model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        match = re.search(r"^model name\s*:\s*(.+)$", cpuinfo.read_text(), re.MULTILINE)
        model = match.group(1) if match else None
    governor = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    return {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cpu_model": model,
        "frequency_governor": governor.read_text().strip() if governor.exists() else None,
        "tools": {
            "g++": version(["g++", "--version"]),
            "rustc": version(["rustc", "--version"]),
            "java": version(["java", "-version"]),
            "cmake": version(["cmake", "--version"]),
        },
    }


def artifact_matches(path: Path) -> int:
    data = path.read_bytes()
    if len(data) < 25 or data[:8] != b"SZLEV001":
        raise RuntimeError(f"invalid result artifact: {path}")
    query_count = struct.unpack_from("=Q", data, 16)[0]
    cursor, matches = 25, 0
    for _ in range(query_count):
        count = struct.unpack_from("=Q", data, cursor)[0]
        cursor += 8 + count * 5
        matches += count
    if cursor != len(data):
        raise RuntimeError(f"malformed result artifact: {path}")
    return matches


def command_for(
    adapter: str,
    args: argparse.Namespace,
    products: dict[str, Path | str],
    queries: Path,
    *,
    validation: bool,
    stringzilla_bound: int | None = None,
) -> tuple[list[str], dict[str, str]]:
    dictionary, limit = str(args.dictionary), str(args.query_count)
    artifacts = args.build_dir / "artifacts"
    if adapter == "stringzilla":
        command = [str(products[adapter]), dictionary, str(queries), limit]
        if validation:
            command.append(str(artifacts / adapter))
        return command, {
            "SZ_LEVENSHTEIN_MAX_DISTANCE": str(stringzilla_bound or 2),
            "SZ_LEVENSHTEIN_REPEATS": "1",
            "SZ_LEVENSHTEIN_MODES": "warm" if validation else args.stringzilla_modes,
        }
    if adapter == "rapidfuzz":
        command = [str(products[adapter]), dictionary, str(queries), limit]
        if validation:
            command.append(str(artifacts / adapter))
        return command, {"RF_MAX_DISTANCE": "2", "RF_REPEATS": "0" if validation else "1", "RF_MODE": "materialized"}
    if adapter in {"symspell", "fst", "tantivy"}:
        environment = {
            "STRINGWARS_FILTER": f"levenshtein/{adapter}",
            "STRINGWARS_REPEATS": "1",
            "STRINGWARS_MIN_DISTANCE": "1",
            "STRINGWARS_MAX_DISTANCE": "2",
        }
        if adapter == "symspell":
            environment["STRINGWARS_SYMSPELL_MODE"] = "exact" if validation else "both"
            if validation:
                environment["STRINGWARS_DUMP_PREFIX"] = str(artifacts / adapter)
        return [str(products["rust"]), dictionary, str(queries), limit], environment
    return [
        "java",
        "-cp",
        str(products["lucene_classpath"]),
        "com.stringzilla.LevenshteinIndexBenchmark",
        dictionary,
        str(queries),
        "2",
        "automaton",
    ], {"STRINGWARS_REPEATS": "1"}


def measured_run(
    adapter: str,
    command: list[str],
    environment: dict[str, str],
    args: argparse.Namespace,
    root: Path,
    log: Path,
) -> dict[str, object]:
    if args.cpu is not None:
        command = ["taskset", "-c", str(args.cpu), *command]
    timing = log.with_suffix(".rss")
    measured = ["/usr/bin/time", "-f", "%M", "-o", str(timing), *command]
    started = time.perf_counter()
    completed = run(measured, cwd=root, env=environment)
    wall = time.perf_counter() - started
    log.write_text(completed.stdout)
    if completed.stderr:
        log.with_suffix(".stderr").write_text(completed.stderr)
    return {
        "adapter": adapter,
        "command": command,
        "environment": environment,
        "log": str(log.relative_to(args.build_dir)),
        "wall_seconds": wall,
        "peak_rss_bytes": int(timing.read_text().strip()) * 1024,
    }


def result_counts(text: str) -> dict[int, set[int]]:
    counts: dict[int, set[int]] = {}
    for line in text.splitlines():
        bound = re.search(r"(?:^| )k=(\d+)", line)
        matches = re.search(r"(?:^| )matches=(\d+)", line)
        if bound and matches:
            counts.setdefault(int(bound.group(1)), set()).add(int(matches.group(1)))
    return counts


def validate(
    args: argparse.Namespace, products: dict[str, Path | str], queries: Path, root: Path, manifest: dict[str, object]
) -> None:
    artifacts = args.build_dir / "artifacts"
    logs = args.build_dir / "logs"
    artifacts.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    validation: dict[str, dict[str, object]] = {}
    for adapter in ADAPTERS:
        if adapter == "stringzilla":
            for bound in (1, 2):
                command, environment = command_for(
                    adapter, args, products, queries, validation=True, stringzilla_bound=bound
                )
                key = f"{adapter}-k{bound}"
                validation[key] = measured_run(adapter, command, environment, args, root, logs / f"validate-{key}.log")
            continue
        command, environment = command_for(adapter, args, products, queries, validation=True)
        entry = measured_run(adapter, command, environment, args, root, logs / f"validate-{adapter}.log")
        validation[adapter] = entry

    expected: dict[int, int] = {}
    for bound in (1, 2):
        files = {
            "stringzilla": artifacts / f"stringzilla.k{bound}.bin",
            "rapidfuzz": artifacts / f"rapidfuzz.k{bound}.bin",
            "symspell": artifacts / f"symspell.symspell.k{bound}.bin",
        }
        reference = files["stringzilla"].read_bytes()
        for adapter, path in files.items():
            if path.read_bytes() != reference:
                raise RuntimeError(f"exact results differ at k={bound}: stringzilla != {adapter}")
        expected[bound] = artifact_matches(files["stringzilla"])
        if expected[bound] == 0:
            raise RuntimeError(f"validation produced no matches at k={bound}")

    for adapter in ("fst", "tantivy", "lucene"):
        observed = result_counts((args.build_dir / str(validation[adapter]["log"])).read_text())
        for bound, count in expected.items():
            if observed.get(bound) != {count}:
                raise RuntimeError(f"{adapter} returned {observed.get(bound)} at k={bound}; expected {count}")
    manifest["expected_matches"] = expected
    manifest["validation"] = validation


def main() -> int:
    args = arguments()
    root = Path(__file__).resolve().parent.parent
    args.build_dir.mkdir(parents=True, exist_ok=True)
    words = args.dictionary.read_text().splitlines()
    if not words or len(words) != len(set(words)) or any(not word.isascii() or word.lower() != word for word in words):
        raise RuntimeError("the shared adapter track requires a non-empty, unique, lowercase ASCII dictionary")

    products = compile_products(args, root)
    queries = args.build_dir / "queries.txt"
    run(
        [str(products["queries"]), str(args.dictionary), str(queries), str(args.query_count), "mixed", str(args.seed)],
        cwd=root,
    )
    manifest: dict[str, object] = {
        "schema": 1,
        "started_at": datetime.now(UTC).isoformat(),
        "machine": {**machine(), "pinned_cpu": args.cpu},
        "inputs": {
            "dictionary": str(args.dictionary),
            "dictionary_sha256": hash_file(args.dictionary),
            "queries": str(queries),
            "queries_sha256": hash_file(queries),
            "query_count": args.query_count,
            "seed": args.seed,
        },
        "revisions": {
            "stringwars": git_revision(root),
            "stringzilla": git_revision(args.stringzilla_root),
            "rapidfuzz": git_revision(args.rapidfuzz_root),
            "cargo_lock_sha256": hash_file(root / "Cargo.lock"),
            "lucene_pom_sha256": hash_file(root / "levenshtein/lucene/pom.xml"),
        },
        "repeats": args.repeats,
        "runs": [],
    }
    output = args.build_dir / "manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        validate(args, products, queries, root, manifest)

        if not args.validation_only:
            randomizer = random.Random(args.seed)
            logs = args.build_dir / "logs"
            for repetition in range(args.repeats):
                order = list(ADAPTERS)
                randomizer.shuffle(order)
                for position, adapter in enumerate(order):
                    bounds = (1, 2) if adapter == "stringzilla" else (None,)
                    for bound in bounds:
                        command, environment = command_for(
                            adapter, args, products, queries, validation=False, stringzilla_bound=bound
                        )
                        suffix = f"-k{bound}" if bound else ""
                        entry = measured_run(
                            adapter,
                            command,
                            environment,
                            args,
                            root,
                            logs / f"run-{repetition:02d}-{position}-{adapter}{suffix}.log",
                        )
                        entry.update({"repetition": repetition, "position": position, "bound": bound})
                        manifest["runs"].append(entry)
                    output.write_text(json.dumps(manifest, indent=2) + "\n")
        manifest["completed_at"] = datetime.now(UTC).isoformat()
    except Exception as error:
        manifest["error"] = str(error)
        output.write_text(json.dumps(manifest, indent=2) + "\n")
        raise
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
