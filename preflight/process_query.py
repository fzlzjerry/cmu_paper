#!/usr/bin/env python3
"""Fail-closed NVIDIA GPU process snapshot for the E00 preflight gate.

The utility deliberately uses only absolute ``/usr/bin/nvidia-smi`` command
paths and Linux ``/proc`` identities.  Executable names are evidence, never an
authorization mechanism: a compute process is supervised only when its PID
and process start time are the supplied root identity, or when its live
``/proc`` ancestry reaches that exact identity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


NVIDIA_SMI = "/usr/bin/nvidia-smi"
GPU_MAP_ARGV = (
    NVIDIA_SMI,
    "--query-gpu=index,uuid",
    "--format=csv,noheader,nounits",
)
COMPUTE_APPS_ARGV = (
    NVIDIA_SMI,
    "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
    "--format=csv,noheader,nounits",
)
PMON_ARGV = (NVIDIA_SMI, "pmon", "-c", "1")

GPU_UUID_RE = re.compile(r"^GPU-[0-9A-Fa-f-]+$")
PMON_PROCESS_TYPES = frozenset({"G", "C", "C+G"})
NULL_MEMORY_VALUES = frozenset(
    {"", "-", "n/a", "[n/a]", "not supported", "[not supported]"}
)
COMMAND_TIMEOUT_SECONDS = 15


class SnapshotError(RuntimeError):
    """An error that must make the process snapshot fail closed."""


class JsonArgumentParser(argparse.ArgumentParser):
    """Argument parser whose failures can still produce the JSON contract."""

    def error(self, message: str) -> None:
        raise SnapshotError(f"argument error: {message}")


@dataclass(frozen=True)
class CommandResult:
    name: str
    argv: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str

    def as_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "argv": list(self.argv),
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_time_ticks: int
    parent_pid: int


@dataclass(frozen=True)
class ComputeAppRow:
    gpu_uuid: str
    pid: int
    process_name: str
    used_gpu_memory_mib: int | float | None


@dataclass(frozen=True)
class PmonRow:
    gpu_index: int
    pid: int
    process_type: str
    process_name: str


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _run_command(name: str, argv: tuple[str, ...]) -> CommandResult:
    env = os.environ.copy()
    env.update({"LC_ALL": "C", "LANG": "C", "TZ": "UTC"})
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
            env=env,
        )
        return CommandResult(
            name=name,
            argv=argv,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        suffix = f"command timed out after {COMMAND_TIMEOUT_SECONDS} seconds"
        stderr = f"{stderr.rstrip()}\n{suffix}" if stderr else suffix
        return CommandResult(name, argv, 124, stdout, stderr)
    except OSError as exc:
        return CommandResult(name, argv, 127, "", f"{type(exc).__name__}: {exc}")


def _require_success(result: CommandResult) -> None:
    if result.exit_code != 0:
        raise SnapshotError(
            f"{result.name} exited with status {result.exit_code}"
        )


def _parse_gpu_map(stdout: str) -> dict[int, str]:
    gpu_map: dict[int, str] = {}
    uuid_to_index: dict[str, int] = {}
    rows = list(
        csv.reader(
            (line for line in stdout.splitlines() if line.strip()),
            skipinitialspace=True,
        )
    )
    if not rows:
        raise SnapshotError("gpu_index_uuid returned no GPU rows")

    for line_number, row in enumerate(rows, start=1):
        if len(row) != 2:
            raise SnapshotError(
                f"gpu_index_uuid line {line_number} has {len(row)} fields, expected 2"
            )
        index_text, gpu_uuid = (field.strip() for field in row)
        try:
            gpu_index = int(index_text)
        except ValueError as exc:
            raise SnapshotError(
                f"gpu_index_uuid line {line_number} has invalid GPU index"
            ) from exc
        if gpu_index < 0:
            raise SnapshotError(
                f"gpu_index_uuid line {line_number} has negative GPU index"
            )
        if not GPU_UUID_RE.fullmatch(gpu_uuid):
            raise SnapshotError(
                f"gpu_index_uuid line {line_number} has invalid GPU UUID"
            )
        if gpu_index in gpu_map:
            raise SnapshotError(f"duplicate GPU index {gpu_index} in gpu_index_uuid")
        if gpu_uuid in uuid_to_index:
            raise SnapshotError(f"duplicate GPU UUID {gpu_uuid} in gpu_index_uuid")
        gpu_map[gpu_index] = gpu_uuid
        uuid_to_index[gpu_uuid] = gpu_index
    return gpu_map


def _parse_pid(text: str, *, source: str, line_number: int) -> int:
    try:
        pid = int(text)
    except ValueError as exc:
        raise SnapshotError(f"{source} line {line_number} has invalid PID") from exc
    if pid <= 0:
        raise SnapshotError(f"{source} line {line_number} has non-positive PID")
    return pid


def _parse_memory_mib(
    text: str, *, source: str, line_number: int
) -> int | float | None:
    normalized = text.strip()
    if normalized.lower() in NULL_MEMORY_VALUES:
        return None
    try:
        value = float(normalized)
    except ValueError as exc:
        raise SnapshotError(
            f"{source} line {line_number} has invalid used GPU memory"
        ) from exc
    if not math.isfinite(value) or value < 0:
        raise SnapshotError(
            f"{source} line {line_number} has negative or non-finite GPU memory"
        )
    return int(value) if value.is_integer() else value


def _parse_compute_apps(stdout: str) -> dict[tuple[str, int], ComputeAppRow]:
    processes: dict[tuple[str, int], ComputeAppRow] = {}
    nonempty_lines = (line for line in stdout.splitlines() if line.strip())
    for line_number, row in enumerate(
        csv.reader(nonempty_lines, skipinitialspace=True), start=1
    ):
        if len(row) != 4:
            raise SnapshotError(
                f"compute_apps line {line_number} has {len(row)} fields, expected 4"
            )
        gpu_uuid, pid_text, process_name, memory_text = (
            field.strip() for field in row
        )
        if not GPU_UUID_RE.fullmatch(gpu_uuid):
            raise SnapshotError(f"compute_apps line {line_number} has invalid GPU UUID")
        pid = _parse_pid(pid_text, source="compute_apps", line_number=line_number)
        if not process_name:
            raise SnapshotError(
                f"compute_apps line {line_number} has an empty process name"
            )
        memory = _parse_memory_mib(
            memory_text, source="compute_apps", line_number=line_number
        )
        key = (gpu_uuid, pid)
        if key in processes:
            raise SnapshotError(
                f"duplicate compute_apps record for GPU {gpu_uuid} PID {pid}"
            )
        processes[key] = ComputeAppRow(gpu_uuid, pid, process_name, memory)
    return processes


def _parse_pmon(stdout: str) -> dict[tuple[int, int], PmonRow]:
    processes: dict[tuple[int, int], PmonRow] = {}
    for line_number, raw_line in enumerate(stdout.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split(maxsplit=9)
        if len(fields) != 10:
            raise SnapshotError(
                f"pmon line {line_number} has {len(fields)} fields, expected 10"
            )
        gpu_text, pid_text, process_type = fields[:3]
        process_name = fields[9].strip()
        try:
            gpu_index = int(gpu_text)
        except ValueError as exc:
            raise SnapshotError(f"pmon line {line_number} has invalid GPU index") from exc
        if gpu_index < 0:
            raise SnapshotError(f"pmon line {line_number} has negative GPU index")
        pid = _parse_pid(pid_text, source="pmon", line_number=line_number)
        if not process_name or process_name == "-":
            raise SnapshotError(f"pmon line {line_number} has no process name")
        key = (gpu_index, pid)
        if key in processes:
            raise SnapshotError(
                f"duplicate pmon record for GPU index {gpu_index} PID {pid}"
            )
        processes[key] = PmonRow(gpu_index, pid, process_type, process_name)
    return processes


def _read_process_identity(pid: int) -> ProcessIdentity:
    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        stat_text = stat_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SnapshotError(f"cannot read {stat_path}: {type(exc).__name__}: {exc}") from exc

    closing_paren = stat_text.rfind(")")
    if closing_paren < 0 or closing_paren + 2 > len(stat_text):
        raise SnapshotError(f"cannot parse {stat_path}: malformed comm field")
    prefix = stat_text[: stat_text.find(" ")]
    tail = stat_text[closing_paren + 2 :].split()
    if len(tail) < 20:
        raise SnapshotError(f"cannot parse {stat_path}: too few fields")
    try:
        stat_pid = int(prefix)
        parent_pid = int(tail[1])
        start_time_ticks = int(tail[19])
    except (ValueError, IndexError) as exc:
        raise SnapshotError(f"cannot parse {stat_path}: invalid numeric field") from exc
    if stat_pid != pid or parent_pid < 0 or start_time_ticks < 0:
        raise SnapshotError(f"cannot parse {stat_path}: inconsistent process identity")
    return ProcessIdentity(pid, start_time_ticks, parent_pid)


def _supervised_relationship(
    process: ProcessIdentity,
    root: ProcessIdentity,
) -> str | None:
    if (
        process.pid == root.pid
        and process.start_time_ticks == root.start_time_ticks
    ):
        return "supervised_child"

    visited = {process.pid}
    ancestor_pid = process.parent_pid
    while ancestor_pid > 0:
        if ancestor_pid in visited:
            raise SnapshotError(
                f"cycle detected while resolving ancestry for PID {process.pid}"
            )
        visited.add(ancestor_pid)
        ancestor = _read_process_identity(ancestor_pid)
        if ancestor.pid == root.pid:
            if ancestor.start_time_ticks == root.start_time_ticks:
                return "supervised_descendant"
            return None
        ancestor_pid = ancestor.parent_pid
    return None


def _observed_record(
    *,
    gpu_uuid: str,
    pid: int,
    start_time_ticks: int,
    process_type: str,
    process_name: str,
    used_gpu_memory_mib: int | float | None,
) -> dict[str, object]:
    return {
        "gpu_uuid": gpu_uuid,
        "pid": pid,
        "process_start_time_ticks": start_time_ticks,
        "process_type": process_type,
        "process_name": process_name,
        "used_gpu_memory_mib": used_gpu_memory_mib,
    }


def _unknown_record(
    *,
    gpu_uuid: str,
    pid: int,
    process_name: str,
    used_gpu_memory_mib: int | float | None,
    start_time_ticks: int = 0,
) -> dict[str, object]:
    return _observed_record(
        gpu_uuid=gpu_uuid,
        pid=pid,
        start_time_ticks=start_time_ticks,
        process_type="UNKNOWN",
        process_name=process_name or "<unknown>",
        used_gpu_memory_mib=used_gpu_memory_mib,
    )


def _classify_processes(
    gpu_map: dict[int, str],
    compute_apps: dict[tuple[str, int], ComputeAppRow],
    pmon: dict[tuple[int, int], PmonRow],
    supervised_root: ProcessIdentity | None,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[str],
]:
    graphics: list[dict[str, object]] = []
    allowed_compute: list[dict[str, object]] = []
    foreign_compute: list[dict[str, object]] = []
    unknown: list[dict[str, object]] = []
    errors: list[str] = []

    pmon_by_uuid_pid: dict[tuple[str, int], PmonRow] = {}
    for (gpu_index, pid), row in sorted(pmon.items()):
        gpu_uuid = gpu_map.get(gpu_index)
        if gpu_uuid is None:
            errors.append(
                f"pmon PID {pid} references unmapped GPU index {gpu_index}"
            )
            continue
        key = (gpu_uuid, pid)
        if key in pmon_by_uuid_pid:
            errors.append(f"pmon aliases duplicate GPU UUID {gpu_uuid} PID {pid}")
            continue
        pmon_by_uuid_pid[key] = row

    all_keys = sorted(set(compute_apps) | set(pmon_by_uuid_pid))
    for gpu_uuid, pid in all_keys:
        compute_row = compute_apps.get((gpu_uuid, pid))
        pmon_row = pmon_by_uuid_pid.get((gpu_uuid, pid))
        process_name = (
            compute_row.process_name
            if compute_row is not None
            else pmon_row.process_name if pmon_row is not None else "<unknown>"
        )
        used_memory = (
            compute_row.used_gpu_memory_mib if compute_row is not None else None
        )

        if gpu_uuid not in gpu_map.values():
            errors.append(
                f"compute_apps PID {pid} references unmapped GPU UUID {gpu_uuid}"
            )
            unknown.append(
                _unknown_record(
                    gpu_uuid=gpu_uuid,
                    pid=pid,
                    process_name=process_name,
                    used_gpu_memory_mib=used_memory,
                )
            )
            continue

        if pmon_row is None:
            errors.append(
                f"compute_apps GPU {gpu_uuid} PID {pid} has no pmon process type"
            )
            try:
                identity = _read_process_identity(pid)
                start_ticks = identity.start_time_ticks
            except SnapshotError as exc:
                errors.append(str(exc))
                start_ticks = 0
            unknown.append(
                _unknown_record(
                    gpu_uuid=gpu_uuid,
                    pid=pid,
                    process_name=process_name,
                    used_gpu_memory_mib=used_memory,
                    start_time_ticks=start_ticks,
                )
            )
            continue

        if pmon_row.process_type not in PMON_PROCESS_TYPES:
            errors.append(
                f"GPU {gpu_uuid} PID {pid} has unknown pmon process type "
                f"{pmon_row.process_type!r}"
            )
            try:
                identity = _read_process_identity(pid)
                start_ticks = identity.start_time_ticks
            except SnapshotError as exc:
                errors.append(str(exc))
                start_ticks = 0
            unknown.append(
                _unknown_record(
                    gpu_uuid=gpu_uuid,
                    pid=pid,
                    process_name=process_name,
                    used_gpu_memory_mib=used_memory,
                    start_time_ticks=start_ticks,
                )
            )
            continue

        if pmon_row.process_type == "G" and compute_row is not None:
            errors.append(
                f"GPU {gpu_uuid} PID {pid} is compute-active but pmon reports G only"
            )
            try:
                identity = _read_process_identity(pid)
                start_ticks = identity.start_time_ticks
            except SnapshotError as exc:
                errors.append(str(exc))
                start_ticks = 0
            unknown.append(
                _unknown_record(
                    gpu_uuid=gpu_uuid,
                    pid=pid,
                    process_name=process_name,
                    used_gpu_memory_mib=used_memory,
                    start_time_ticks=start_ticks,
                )
            )
            continue

        try:
            identity = _read_process_identity(pid)
        except SnapshotError as exc:
            errors.append(str(exc))
            unknown.append(
                _unknown_record(
                    gpu_uuid=gpu_uuid,
                    pid=pid,
                    process_name=process_name,
                    used_gpu_memory_mib=used_memory,
                )
            )
            continue

        if pmon_row.process_type == "G":
            graphics.append(
                _observed_record(
                    gpu_uuid=gpu_uuid,
                    pid=pid,
                    start_time_ticks=identity.start_time_ticks,
                    process_type="G",
                    process_name=process_name,
                    used_gpu_memory_mib=used_memory,
                )
            )
            continue

        base_record = _observed_record(
            gpu_uuid=gpu_uuid,
            pid=pid,
            start_time_ticks=identity.start_time_ticks,
            process_type=pmon_row.process_type,
            process_name=process_name,
            used_gpu_memory_mib=used_memory,
        )
        if supervised_root is None:
            foreign_compute.append(base_record)
            continue
        try:
            relationship = _supervised_relationship(identity, supervised_root)
        except SnapshotError as exc:
            errors.append(str(exc))
            unknown.append(
                _unknown_record(
                    gpu_uuid=gpu_uuid,
                    pid=pid,
                    process_name=process_name,
                    used_gpu_memory_mib=used_memory,
                    start_time_ticks=identity.start_time_ticks,
                )
            )
            continue
        if relationship is None:
            foreign_compute.append(base_record)
            continue

        allowed_record = dict(base_record)
        allowed_record.update(
            {
                "relationship": relationship,
                "supervised_root_identity": {
                    "pid": supervised_root.pid,
                    "start_time_ticks": supervised_root.start_time_ticks,
                },
            }
        )
        allowed_compute.append(allowed_record)

    sort_key = lambda item: (str(item["gpu_uuid"]), int(item["pid"]))
    for records in (graphics, allowed_compute, foreign_compute, unknown):
        records.sort(key=sort_key)
    return graphics, allowed_compute, foreign_compute, unknown, errors


def _empty_output(captured_at_utc: str, error: str) -> dict[str, object]:
    return {
        "captured_at_utc": captured_at_utc,
        "query_exit_code": 2,
        "graphics_processes": [],
        "allowed_compute_processes": [],
        "foreign_compute_processes": [],
        "unknown_processes": [],
        "subcommands": [],
        "errors": [error],
    }


def _parse_nonnegative_integer(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def _parse_positive_pid(text: str) -> int:
    value = _parse_nonnegative_integer(text)
    if value == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = JsonArgumentParser(
        description="Capture a fail-closed E00 NVIDIA GPU process snapshot."
    )
    parser.add_argument("--supervised-root-pid", type=_parse_positive_pid)
    parser.add_argument(
        "--supervised-root-start-ticks", type=_parse_nonnegative_integer
    )
    args = parser.parse_args(argv)
    paired = (
        args.supervised_root_pid is not None,
        args.supervised_root_start_ticks is not None,
    )
    if paired[0] != paired[1]:
        raise SnapshotError(
            "--supervised-root-pid and --supervised-root-start-ticks must be supplied together"
        )
    return args


def _snapshot(args: argparse.Namespace) -> tuple[dict[str, object], int]:
    captured_at_utc = _utc_now()
    command_results = [
        _run_command("gpu_index_uuid", GPU_MAP_ARGV),
        _run_command("compute_apps", COMPUTE_APPS_ARGV),
        _run_command("pmon", PMON_ARGV),
    ]
    errors: list[str] = []
    for result in command_results:
        try:
            _require_success(result)
        except SnapshotError as exc:
            errors.append(str(exc))

    gpu_map: dict[int, str] = {}
    compute_apps: dict[tuple[str, int], ComputeAppRow] = {}
    pmon: dict[tuple[int, int], PmonRow] = {}
    if not errors:
        for parser, stdout in (
            (_parse_gpu_map, command_results[0].stdout),
            (_parse_compute_apps, command_results[1].stdout),
            (_parse_pmon, command_results[2].stdout),
        ):
            try:
                parsed = parser(stdout)
                if parser is _parse_gpu_map:
                    gpu_map = parsed
                elif parser is _parse_compute_apps:
                    compute_apps = parsed
                else:
                    pmon = parsed
            except SnapshotError as exc:
                errors.append(str(exc))

    supervised_root: ProcessIdentity | None = None
    if args.supervised_root_pid is not None:
        try:
            candidate = _read_process_identity(args.supervised_root_pid)
            if candidate.start_time_ticks != args.supervised_root_start_ticks:
                raise SnapshotError(
                    "supervised root PID exists but its process start time does not match"
                )
            supervised_root = candidate
        except SnapshotError as exc:
            errors.append(str(exc))

    graphics: list[dict[str, object]] = []
    allowed: list[dict[str, object]] = []
    foreign: list[dict[str, object]] = []
    unknown: list[dict[str, object]] = []
    if not errors:
        graphics, allowed, foreign, unknown, classification_errors = (
            _classify_processes(gpu_map, compute_apps, pmon, supervised_root)
        )
        errors.extend(classification_errors)

    query_exit_code = 0
    if errors:
        nonzero_codes = [
            result.exit_code for result in command_results if result.exit_code != 0
        ]
        query_exit_code = nonzero_codes[0] if nonzero_codes else 2

    output = {
        "captured_at_utc": captured_at_utc,
        "query_exit_code": query_exit_code,
        "graphics_processes": graphics,
        "allowed_compute_processes": allowed,
        "foreign_compute_processes": foreign,
        "unknown_processes": unknown,
        "subcommands": [result.as_json() for result in command_results],
        "errors": errors,
    }
    return output, query_exit_code


def main(argv: Sequence[str] | None = None) -> int:
    captured_at_utc = _utc_now()
    try:
        args = _parse_args(sys.argv[1:] if argv is None else argv)
        output, exit_code = _snapshot(args)
    except SnapshotError as exc:
        output = _empty_output(captured_at_utc, str(exc))
        exit_code = 2
    except Exception as exc:  # Defensive: the gate must fail closed with evidence.
        output = _empty_output(
            captured_at_utc, f"unexpected {type(exc).__name__}: {exc}"
        )
        exit_code = 2
    json.dump(output, sys.stdout, sort_keys=True, separators=(",", ":"))
    sys.stdout.write("\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
