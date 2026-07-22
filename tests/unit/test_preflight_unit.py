"""Pure unit checks for the E00 collector contract."""

from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import jsonschema

from preflight import process_query
from preflight import run_preflight
from preflight.audit_checkpoint import audit_checkpoint


ROOT = Path(__file__).resolve().parents[2]


class PreflightUnitTests(unittest.TestCase):
    def test_gate_order_matches_schema(self) -> None:
        schema = json.loads(
            (ROOT / "preflight" / "e00_manifest.schema.json").read_text()
        )
        prefix = schema["$defs"]["gate"]["properties"]["checks"]["prefixItems"]
        observed = [
            item["allOf"][1]["properties"]["name"]["const"] for item in prefix
        ]
        self.assertEqual(observed, list(run_preflight.GATE_NAMES))

    def test_schema_is_draft_2020_12(self) -> None:
        schema = json.loads(
            (ROOT / "preflight" / "e00_manifest.schema.json").read_text()
        )
        jsonschema.Draft202012Validator.check_schema(schema)

    def test_python_lock_is_hash_complete(self) -> None:
        locked = run_preflight.parse_requirements_lock(
            ROOT / "preflight" / "requirements-e00.txt"
        )
        self.assertEqual(locked["torch"], "2.12.1+cu130")
        self.assertEqual(locked["triton"], "3.7.1")
        self.assertEqual(locked["jsonschema"], "4.25.1")
        self.assertGreaterEqual(len(locked), 30)

    def test_gpu_csv_parser_rejects_missing_field(self) -> None:
        malformed = ",".join(["x"] * (len(run_preflight.GPU_QUERY_FIELDS) - 1))
        with self.assertRaises(ValueError):
            run_preflight.parse_gpu_rows(malformed)

    def test_compute_capability_is_canonical(self) -> None:
        self.assertEqual(
            run_preflight.capability("12.0"),
            {"major": 12, "minor": 0, "text": "12.0"},
        )
        self.assertIsNone(run_preflight.capability("sm_120"))

    def test_numeric_and_torch_payload_normalizers_fail_closed(self) -> None:
        for value in (
            None,
            "",
            "N/A",
            "[N/A]",
            "Unknown",
            "Not Supported",
            "garbage",
            "nan",
            "inf",
            "-inf",
        ):
            self.assertIsNone(run_preflight.nullable_float(value))
        self.assertEqual(run_preflight.nullable_float(" 1.25 "), 1.25)
        self.assertEqual(run_preflight.nullable_nonnegative_float("0"), 0.0)
        self.assertIsNone(run_preflight.nullable_nonnegative_float("-0.1"))

        self.assertEqual(run_preflight.nullable_int("3.0"), 3)
        for value in ("3.2", "-1", "nan", "not-an-int"):
            self.assertIsNone(run_preflight.nullable_int(value))

        self.assertEqual(run_preflight.nonnegative_int_value(2), 2)
        self.assertEqual(
            run_preflight.nonnegative_int_value(2, minimum=2), 2
        )
        for value in (True, False, 1.0, "1", -1, None):
            self.assertIsNone(run_preflight.nonnegative_int_value(value))
        self.assertIsNone(
            run_preflight.nonnegative_int_value(1, minimum=2)
        )

        self.assertIs(run_preflight.boolean_value(True), True)
        self.assertIs(run_preflight.boolean_value(False), False)
        for value in (0, 1, "true", None):
            self.assertIsNone(run_preflight.boolean_value(value))

        self.assertEqual(
            run_preflight.capability_from_mapping(
                {"major": 12, "minor": 0}
            ),
            {"major": 12, "minor": 0, "text": "12.0"},
        )
        for value in (
            None,
            [],
            {"major": True, "minor": 0},
            {"major": 0, "minor": 0},
            {"major": 12.0, "minor": 0},
            {"major": "12", "minor": 0},
            {"major": 12, "minor": -1},
        ):
            self.assertIsNone(run_preflight.capability_from_mapping(value))

        self.assertEqual(run_preflight.nonempty_string("  torch  "), "torch")
        for value in (None, "", "   ", 12, False):
            self.assertIsNone(run_preflight.nonempty_string(value))
        self.assertEqual(
            run_preflight.matching_string(" GPU-abCD-1234 ", r"GPU-[\w-]+"),
            "GPU-abCD-1234",
        )
        self.assertIsNone(
            run_preflight.matching_string("not-a-gpu", r"GPU-[\w-]+")
        )

    def test_process_query_parses_graphics_type(self) -> None:
        parsed = process_query._parse_pmon(
            "0 1404 G - - - - - - Xorg\n"
        )
        row = parsed[(0, 1404)]
        self.assertEqual(row.process_type, "G")
        self.assertEqual(row.process_name, "Xorg")

    def test_process_query_rejects_unknown_uuid(self) -> None:
        with self.assertRaises(process_query.SnapshotError):
            process_query._parse_compute_apps(
                "not-a-gpu, 99, python, 1\n"
            )

    def test_environment_allowlist_matches_schema_exactly(self) -> None:
        schema = json.loads(
            (ROOT / "preflight" / "e00_manifest.schema.json").read_text()
        )
        command_environment = schema["$defs"]["commandEnvironment"]
        self.assertEqual(
            set(command_environment["properties"]),
            set(run_preflight.ENVIRONMENT_ALLOWLIST),
        )
        self.assertTrue(
            {
                "LC_ALL",
                "LANG",
                "TZ",
                "PATH",
                "CUDA_HOME",
                "PYTHONNOUSERSITE",
                "PYTHONOPTIMIZE",
                "PYTHONHASHSEED",
            }.issubset(command_environment["required"])
        )
        self.assertEqual(
            command_environment["properties"]["PYTHONOPTIMIZE"]["const"], "0"
        )

    def test_torch_probe_producer_keys_match_schema_exactly(self) -> None:
        schema = json.loads(
            (ROOT / "preflight" / "e00_manifest.schema.json").read_text()
        )
        expected = set(schema["$defs"]["torchProbe"]["properties"])
        tree = ast.parse(
            (ROOT / "preflight" / "run_preflight.py").read_text()
        )
        producer: ast.Dict | None = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if any(
                isinstance(target, ast.Name)
                and target.id == "torch_probe_manifest"
                for target in node.targets
            ):
                self.assertIsInstance(node.value, ast.Dict)
                producer = node.value
                break
        self.assertIsNotNone(producer)
        if producer is None:
            self.fail("torch_probe_manifest producer was not found")
        observed = {
            key.value
            for key in producer.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        self.assertEqual(observed, expected)

    def test_cuobjdump_ids_are_conditional_on_inspection(self) -> None:
        schema = json.loads(
            (ROOT / "preflight" / "e00_manifest.schema.json").read_text()
        )
        extension = schema["$defs"]["extension"]
        fragment = {
            "$defs": schema["$defs"],
            "type": "object",
            "required": ["binary_inspection", "cuobjdump_command_ids"],
            "properties": {
                "binary_inspection": {"type": "object"},
                "cuobjdump_command_ids": extension["properties"][
                    "cuobjdump_command_ids"
                ],
            },
            "allOf": extension["allOf"],
        }
        validator = jsonschema.Draft202012Validator(fragment)
        self.assertFalse(
            list(
                validator.iter_errors(
                    {
                        "binary_inspection": {"status": "NOT_RUN"},
                        "cuobjdump_command_ids": [],
                    }
                )
            )
        )
        self.assertTrue(
            list(
                validator.iter_errors(
                    {
                        "binary_inspection": {"status": "NOT_RUN"},
                        "cuobjdump_command_ids": ["fabricated"],
                    }
                )
            )
        )
        self.assertTrue(
            list(
                validator.iter_errors(
                    {
                        "binary_inspection": {"status": "PASS"},
                        "cuobjdump_command_ids": [],
                    }
                )
            )
        )
        self.assertFalse(
            list(
                validator.iter_errors(
                    {
                        "binary_inspection": {"status": "PASS"},
                        "cuobjdump_command_ids": ["one", "two"],
                    }
                )
            )
        )

    def test_torch_probe_python_lock_mismatch_requires_collection_fail(
        self,
    ) -> None:
        schema = json.loads(
            (ROOT / "preflight" / "e00_manifest.schema.json").read_text()
        )
        validator = jsonschema.Draft202012Validator(
            {
                "$defs": schema["$defs"],
                "$ref": "#/$defs/torchProbe",
            }
        )
        mismatch = {
            "collection_status": "FAIL",
            "collection_error": "locked Python executable mismatch",
            "cuda_available": True,
            "device_count": 1,
            "selected_logical_index": 0,
            "selected_device_name": "NVIDIA RTX PRO 6000 Blackwell",
            "python_optimization_level": 0,
            "python_executable": "/unexpected/python",
            "python_executable_matches_lock": False,
            "selected_device_uuid": "GPU-abcd-1234",
            "selected_device_capability": {
                "major": 12,
                "minor": 0,
                "text": "12.0",
            },
            "selected_device_total_memory_bytes": 1,
            "selected_uuid_matches_manifest": True,
            "capability_matches_manifest": True,
        }
        self.assertFalse(list(validator.iter_errors(mismatch)))

        invalid_pass = copy.deepcopy(mismatch)
        invalid_pass["collection_status"] = "PASS"
        invalid_pass["collection_error"] = None
        self.assertTrue(list(validator.iter_errors(invalid_pass)))

        valid_pass = copy.deepcopy(invalid_pass)
        valid_pass["python_executable_matches_lock"] = True
        self.assertFalse(list(validator.iter_errors(valid_pass)))

    def test_software_failure_observations_remain_schema_valid(self) -> None:
        schema = json.loads(
            (ROOT / "preflight" / "e00_manifest.schema.json").read_text()
        )

        def validator_for(definition: str) -> jsonschema.Draft202012Validator:
            return jsonschema.Draft202012Validator(
                {
                    "$defs": schema["$defs"],
                    "$ref": f"#/$defs/{definition}",
                }
            )

        def package(version: str) -> dict[str, object]:
            return {
                "installed": True,
                "version": version,
                "observation_error": None,
            }

        complete = {
            "collection_status": "PASS",
            "collection_error": None,
            "nvidia_driver": {
                "version": "580.95.05",
                "driver_supported_cuda_version": "13.0",
            },
            "cuda_toolkit": {
                "installed": True,
                "version": "13.0",
                "cuda_home": "/usr/local/cuda-13.0",
                "nvcc_path": "/usr/local/cuda-13.0/bin/nvcc",
                "observation_error": None,
            },
            "ncu": package("2025.3.1"),
            "nsys": package("2025.3.2"),
            "compute_sanitizer": package("2025.3.1"),
            "python": package("3.12.3"),
            "torch": {
                "installed": True,
                "version": "2.12.1+cu130",
                "compiled_cuda_version": "13.0",
                "observation_error": None,
            },
            "triton": package("3.7.1"),
            "vllm": {
                "installed": False,
                "version": None,
                "observation_error": "vLLM not installed",
            },
        }
        software_validator = validator_for("software")
        self.assertFalse(list(software_validator.iter_errors(complete)))

        package_validator = validator_for("packageObservation")
        unparsed_package = {
            "installed": True,
            "version": None,
            "observation_error": "version output was unparsable",
        }
        self.assertFalse(
            list(package_validator.iter_errors(unparsed_package))
        )
        absent_package = {
            "installed": False,
            "version": None,
            "observation_error": "package not installed",
        }
        self.assertFalse(list(package_validator.iter_errors(absent_package)))
        false_with_version = copy.deepcopy(absent_package)
        false_with_version["version"] = "fabricated"
        self.assertTrue(
            list(package_validator.iter_errors(false_with_version))
        )

        failed_unparsed = copy.deepcopy(complete)
        failed_unparsed["collection_status"] = "FAIL"
        failed_unparsed["collection_error"] = "ncu version was unparsable"
        failed_unparsed["ncu"] = unparsed_package
        self.assertFalse(
            list(software_validator.iter_errors(failed_unparsed))
        )
        invalid_unparsed_pass = copy.deepcopy(failed_unparsed)
        invalid_unparsed_pass["collection_status"] = "PASS"
        invalid_unparsed_pass["collection_error"] = None
        self.assertTrue(
            list(software_validator.iter_errors(invalid_unparsed_pass))
        )

        cuda_validator = validator_for("cudaToolkitObservation")
        absent_cuda = {
            "installed": False,
            "version": None,
            "cuda_home": None,
            "nvcc_path": None,
            "observation_error": "nvcc command failed",
        }
        self.assertFalse(list(cuda_validator.iter_errors(absent_cuda)))
        for field, value in (
            ("version", "13.0"),
            ("cuda_home", "/usr/local/cuda-13.0"),
            ("nvcc_path", "/usr/local/cuda-13.0/bin/nvcc"),
        ):
            fabricated_cuda = copy.deepcopy(absent_cuda)
            fabricated_cuda[field] = value
            self.assertTrue(
                list(cuda_validator.iter_errors(fabricated_cuda)),
                field,
            )

        incomplete_cuda = copy.deepcopy(complete)
        incomplete_cuda["collection_status"] = "FAIL"
        incomplete_cuda["collection_error"] = (
            "nvcc version output was unparsable"
        )
        incomplete_cuda["cuda_toolkit"] = {
            "installed": True,
            "version": None,
            "cuda_home": "/usr/local/cuda-13.0",
            "nvcc_path": "/usr/local/cuda-13.0/bin/nvcc",
            "observation_error": "nvcc version unavailable",
        }
        self.assertFalse(
            list(software_validator.iter_errors(incomplete_cuda))
        )
        invalid_cuda_pass = copy.deepcopy(incomplete_cuda)
        invalid_cuda_pass["collection_status"] = "PASS"
        invalid_cuda_pass["collection_error"] = None
        self.assertTrue(
            list(software_validator.iter_errors(invalid_cuda_pass))
        )

        missing_driver = copy.deepcopy(complete)
        missing_driver["collection_status"] = "FAIL"
        missing_driver["collection_error"] = "driver version unavailable"
        missing_driver["nvidia_driver"]["version"] = None
        self.assertFalse(
            list(software_validator.iter_errors(missing_driver))
        )
        invalid_driver_pass = copy.deepcopy(missing_driver)
        invalid_driver_pass["collection_status"] = "PASS"
        invalid_driver_pass["collection_error"] = None
        self.assertTrue(
            list(software_validator.iter_errors(invalid_driver_pass))
        )

        for package_name in (
            "ncu",
            "nsys",
            "compute_sanitizer",
            "python",
            "triton",
        ):
            missing_package = copy.deepcopy(complete)
            missing_package["collection_status"] = "FAIL"
            missing_package["collection_error"] = (
                f"{package_name} unavailable"
            )
            missing_package[package_name] = {
                "installed": False,
                "version": None,
                "observation_error": f"{package_name} not installed",
            }
            self.assertFalse(
                list(software_validator.iter_errors(missing_package)),
                package_name,
            )
            invalid_package_pass = copy.deepcopy(missing_package)
            invalid_package_pass["collection_status"] = "PASS"
            invalid_package_pass["collection_error"] = None
            self.assertTrue(
                list(software_validator.iter_errors(invalid_package_pass)),
                package_name,
            )

        missing_torch_metadata = copy.deepcopy(complete)
        missing_torch_metadata["collection_status"] = "FAIL"
        missing_torch_metadata["collection_error"] = (
            "PyTorch metadata unavailable"
        )
        missing_torch_metadata["torch"] = {
            "installed": True,
            "version": None,
            "compiled_cuda_version": None,
            "observation_error": "PyTorch metadata invalid",
        }
        self.assertFalse(
            list(software_validator.iter_errors(missing_torch_metadata))
        )
        invalid_torch_pass = copy.deepcopy(missing_torch_metadata)
        invalid_torch_pass["collection_status"] = "PASS"
        invalid_torch_pass["collection_error"] = None
        self.assertTrue(
            list(software_validator.iter_errors(invalid_torch_pass))
        )

        missing_vllm = copy.deepcopy(complete)
        del missing_vllm["vllm"]
        self.assertTrue(
            list(software_validator.iter_errors(missing_vllm))
        )
        malformed_vllm = copy.deepcopy(complete)
        malformed_vllm["vllm"] = {
            "installed": "false",
            "version": None,
            "observation_error": "invalid boolean",
        }
        self.assertTrue(
            list(software_validator.iter_errors(malformed_vllm))
        )

    def test_vllm_payload_validity_distinguishes_observed_absence(
        self,
    ) -> None:
        tree = ast.parse(
            (ROOT / "preflight" / "run_preflight.py").read_text()
        )
        function_node = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "payload_package_observation"
        )
        module = ast.fix_missing_locations(
            ast.Module(
                body=[copy.deepcopy(function_node)],
                type_ignores=[],
            )
        )
        compiled = compile(
            module,
            str(ROOT / "preflight" / "run_preflight.py"),
            "exec",
        )

        def observe(
            payload: dict[str, object],
        ) -> tuple[dict[str, object], bool]:
            validity: dict[str, bool] = {}

            def package_observation(
                presence_observed: bool,
                version: str | None,
                error: str,
            ) -> dict[str, object]:
                normalized = run_preflight.nonempty_string(version)
                return {
                    "installed": presence_observed,
                    "version": normalized if presence_observed else None,
                    "observation_error": (
                        None
                        if presence_observed and normalized is not None
                        else error
                    ),
                }

            namespace = {
                "Any": object,
                "boolean_value": run_preflight.boolean_value,
                "nonempty_string": run_preflight.nonempty_string,
                "package_observation": package_observation,
                "parsed_torch_payload": payload,
                "payload_package_validity": validity,
                "torch_payload_parsed": True,
            }
            exec(compiled, namespace)
            observation = namespace["payload_package_observation"]("vllm")
            return observation, validity.get("vllm", False)

        absent, absent_valid = observe(
            {"vllm": {"installed": False, "version": None}}
        )
        self.assertTrue(absent_valid)
        self.assertIs(absent["installed"], False)
        self.assertIsNone(absent["version"])

        present, present_valid = observe(
            {"vllm": {"installed": True, "version": "0.10.2"}}
        )
        self.assertTrue(present_valid)
        self.assertIs(present["installed"], True)
        self.assertEqual(present["version"], "0.10.2")

        malformed_payloads = (
            {},
            {"vllm": None},
            {"vllm": []},
            {"vllm": {"installed": 0, "version": None}},
            {"vllm": {"installed": False, "version": "fabricated"}},
            {"vllm": {"installed": True, "version": None}},
            {"vllm": {"installed": True, "version": "   "}},
        )
        for payload in malformed_payloads:
            _, valid = observe(payload)
            self.assertFalse(valid, payload)

    def test_kernel_image_pass_requires_native_and_forced_ptx(self) -> None:
        tree = ast.parse(
            (ROOT / "preflight" / "run_preflight.py").read_text()
        )

        def assignment(name: str) -> ast.Assign:
            return next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == name
                    for target in node.targets
                )
            )

        proven_expression = assignment("kernel_execution_proven").value
        expected_proven = ast.parse(
            "native_ok and forced_ptx_ok",
            mode="eval",
        ).body
        self.assertEqual(
            ast.dump(proven_expression, include_attributes=False),
            ast.dump(expected_proven, include_attributes=False),
        )

        proven_code = compile(
            ast.fix_missing_locations(
                ast.Expression(body=copy.deepcopy(proven_expression))
            ),
            str(ROOT / "preflight" / "run_preflight.py"),
            "eval",
        )
        status_code = compile(
            ast.fix_missing_locations(
                ast.Expression(
                    body=copy.deepcopy(assignment("kernel_status").value)
                )
            ),
            str(ROOT / "preflight" / "run_preflight.py"),
            "eval",
        )

        def kernel_status(
            *,
            native_ok: bool,
            forced_ptx_ok: bool,
            kernel_image_error: bool,
            any_runtime_ran: bool,
        ) -> str:
            scope = {
                "native_ok": native_ok,
                "forced_ptx_ok": forced_ptx_ok,
            }
            kernel_execution_proven = eval(
                proven_code,
                {"__builtins__": {}},
                scope,
            )
            return eval(
                status_code,
                {"__builtins__": {}},
                {
                    "kernel_execution_proven": kernel_execution_proven,
                    "kernel_image_error": kernel_image_error,
                    "any_runtime_ran": any_runtime_ran,
                },
            )

        self.assertEqual(
            kernel_status(
                native_ok=True,
                forced_ptx_ok=True,
                kernel_image_error=False,
                any_runtime_ran=True,
            ),
            "PASS",
        )
        for native_ok, forced_ptx_ok in (
            (False, False),
            (False, True),
            (True, False),
        ):
            self.assertEqual(
                kernel_status(
                    native_ok=native_ok,
                    forced_ptx_ok=forced_ptx_ok,
                    kernel_image_error=False,
                    any_runtime_ran=True,
                ),
                "FAIL",
            )
        self.assertEqual(
            kernel_status(
                native_ok=True,
                forced_ptx_ok=True,
                kernel_image_error=True,
                any_runtime_ran=True,
            ),
            "FAIL",
        )
        self.assertEqual(
            kernel_status(
                native_ok=False,
                forced_ptx_ok=False,
                kernel_image_error=False,
                any_runtime_ran=False,
            ),
            "NOT_RUN",
        )

    def test_process_audit_initial_only_is_valid_for_fail_not_pass(self) -> None:
        schema = json.loads(
            (ROOT / "preflight" / "e00_manifest.schema.json").read_text()
        )
        pass_process_constraints = schema["allOf"][0]["then"][
            "properties"
        ]["process_audit"]
        fragment = {
            "$defs": schema["$defs"],
            "type": "object",
            "additionalProperties": False,
            "required": ["aggregate_status", "process_audit"],
            "properties": {
                "aggregate_status": {"enum": ["PASS", "FAIL"]},
                "process_audit": {
                    "$ref": "#/$defs/processAuditProducer"
                },
            },
            "allOf": [
                {
                    "if": {
                        "properties": {
                            "aggregate_status": {"const": "PASS"}
                        },
                        "required": ["aggregate_status"],
                    },
                    "then": {
                        "properties": {
                            "process_audit": pass_process_constraints
                        }
                    },
                }
            ],
        }
        validator = jsonschema.Draft202012Validator(fragment)
        output = {
            "path": "commands/process_initial.stdout.txt",
            "sha256": "0" * 64,
            "size_bytes": 0,
        }
        initial_snapshot = {
            "phase": "before",
            "command_id": "process_initial",
            "captured_at_utc": "2026-07-22T00:00:00Z",
            "query_argv": ["unit-process-query"],
            "query_exit_code": 0,
            "raw_stdout": output,
            "raw_stderr": {
                **output,
                "path": "commands/process_initial.stderr.txt",
            },
            "graphics_processes": [],
            "allowed_compute_processes": [],
            "foreign_compute_processes": [],
            "unknown_processes": [],
        }
        process_audit = {
            "policy": {
                "graphics_only_allowed": True,
                "foreign_compute_allowed": False,
                "supervised_identity_basis": "pid_and_process_start_time",
                "executable_name_whitelist_used": False,
                "unknown_process_type_fails_closed": True,
            },
            "collector_identity": {"pid": 1, "start_time_ticks": 0},
            "query_failure_count": 0,
            "graphics_process_count": 0,
            "supervised_compute_process_count": 0,
            "foreign_or_unknown_process_count": 0,
            "snapshots": [initial_snapshot],
        }
        failed_gate = {
            "aggregate_status": "FAIL",
            "process_audit": process_audit,
        }
        self.assertFalse(list(validator.iter_errors(failed_gate)))

        passing_gate = copy.deepcopy(failed_gate)
        passing_gate["aggregate_status"] = "PASS"
        self.assertTrue(list(validator.iter_errors(passing_gate)))

        for phase in ("during", "after"):
            snapshot = copy.deepcopy(initial_snapshot)
            snapshot["phase"] = phase
            snapshot["command_id"] = f"process_unit_{phase}"
            passing_gate["process_audit"]["snapshots"].append(snapshot)
        self.assertFalse(list(validator.iter_errors(passing_gate)))

    def test_not_run_cuda_result_fields_are_truthful(self) -> None:
        schema = json.loads(
            (ROOT / "preflight" / "e00_manifest.schema.json").read_text()
        )

        def validator_for(definition: str) -> jsonschema.Draft202012Validator:
            return jsonschema.Draft202012Validator(
                {
                    "$defs": schema["$defs"],
                    "$ref": f"#/$defs/{definition}",
                }
            )

        envelope = {
            "status": "NOT_RUN",
            "reason": "not admitted",
            "command_ids": [],
            "evidence_file_ids": [],
        }
        native = {
            **envelope,
            "separate_process": None,
            "cuda_disable_ptx_jit": None,
        }
        native_validator = validator_for("nativeExecutionResult")
        self.assertFalse(list(native_validator.iter_errors(native)))
        fabricated_native = copy.deepcopy(native)
        fabricated_native["separate_process"] = True
        self.assertTrue(list(native_validator.iter_errors(fabricated_native)))

        forced_ptx = {
            **envelope,
            "separate_process": None,
            "cuda_force_ptx_jit": None,
            "fresh_cuda_cache": None,
        }
        forced_ptx_validator = validator_for("forcedPtxResult")
        self.assertFalse(list(forced_ptx_validator.iter_errors(forced_ptx)))
        fabricated_ptx = copy.deepcopy(forced_ptx)
        fabricated_ptx["fresh_cuda_cache"] = True
        self.assertTrue(
            list(forced_ptx_validator.iter_errors(fabricated_ptx))
        )

        allocation = {
            **envelope,
            "preallocated_input_output": None,
            "allocated_delta_bytes": None,
            "reserved_delta_bytes": None,
            "output_pointer_stable": None,
        }
        allocation_validator = validator_for("allocationResult")
        self.assertFalse(list(allocation_validator.iter_errors(allocation)))
        fabricated_allocation = copy.deepcopy(allocation)
        fabricated_allocation["preallocated_input_output"] = True
        self.assertTrue(
            list(allocation_validator.iter_errors(fabricated_allocation))
        )

        sanitizer = {
            **envelope,
            "tools_run": [],
            "leak_check_full": None,
            "error_count": None,
        }
        sanitizer_validator = validator_for("sanitizerResult")
        self.assertFalse(list(sanitizer_validator.iter_errors(sanitizer)))
        fabricated_sanitizer = copy.deepcopy(sanitizer)
        fabricated_sanitizer.update(
            {
                "tools_run": [
                    "memcheck",
                    "initcheck",
                    "racecheck",
                    "synccheck",
                ],
                "leak_check_full": True,
            }
        )
        self.assertTrue(
            list(sanitizer_validator.iter_errors(fabricated_sanitizer))
        )

        partial_sanitizer_failure = {
            **envelope,
            "status": "FAIL",
            "reason": "memcheck failed",
            "command_ids": ["sanitizer_memcheck"],
            "tools_run": ["memcheck"],
            "leak_check_full": True,
            "error_count": 1,
        }
        self.assertFalse(
            list(
                sanitizer_validator.iter_errors(partial_sanitizer_failure)
            )
        )

        not_run_results = (
            (
                "numericalGoldenResult",
                {
                    **envelope,
                    "dtype": "int32",
                    "case_count": None,
                    "atol": None,
                    "rtol": None,
                    "max_abs_error": None,
                    "max_rel_error": None,
                },
                {
                    "case_count": 1,
                    "atol": 0.0,
                    "rtol": 0.0,
                    "max_abs_error": 0.0,
                    "max_rel_error": 0.0,
                },
            ),
            (
                "graphResult",
                {
                    **envelope,
                    "capture_succeeded": None,
                    "replay_count": None,
                    "output_matches_golden": None,
                },
                {
                    "capture_succeeded": True,
                    "replay_count": 1,
                    "output_matches_golden": True,
                },
            ),
            (
                "binaryInspectionResult",
                {
                    **envelope,
                    "sass_target_present": None,
                    "ptx_target_present": None,
                },
                {
                    "sass_target_present": True,
                    "ptx_target_present": True,
                },
            ),
            (
                "kernelImageResult",
                {
                    **envelope,
                    "error_found": None,
                },
                {
                    "error_found": False,
                },
            ),
        )
        for definition, truthful, fabricated_values in not_run_results:
            result_validator = validator_for(definition)
            self.assertFalse(
                list(result_validator.iter_errors(truthful)),
                definition,
            )
            for field, value in fabricated_values.items():
                fabricated = copy.deepcopy(truthful)
                fabricated[field] = value
                self.assertTrue(
                    list(result_validator.iter_errors(fabricated)),
                    f"{definition}.{field}",
                )
            for id_field in ("command_ids", "evidence_file_ids"):
                fabricated_ids = copy.deepcopy(truthful)
                fabricated_ids[id_field] = ["prerequisite_evidence"]
                self.assertTrue(
                    list(result_validator.iter_errors(fabricated_ids)),
                    f"{definition}.{id_field}",
                )

        generic_not_run = {
            "status": "NOT_RUN",
            "reason": "prerequisite gate failed",
            "command_ids": ["prerequisite_inventory"],
            "evidence_file_ids": ["prerequisite_inventory_stdout"],
        }
        self.assertFalse(
            list(
                validator_for("resultEnvelope").iter_errors(
                    generic_not_run
                )
            )
        )
        generic_gate = {
            "name": "extension_build",
            **generic_not_run,
        }
        self.assertFalse(
            list(validator_for("gateCheck").iter_errors(generic_gate))
        )

    def test_uninspected_architecture_has_no_fabricated_targets(self) -> None:
        schema = json.loads(
            (ROOT / "preflight" / "e00_manifest.schema.json").read_text()
        )
        extension = schema["$defs"]["extension"]
        fragment = {
            "$defs": schema["$defs"],
            "type": "object",
            "additionalProperties": False,
            "required": [
                "architecture",
                "binary_inspection",
                "cuobjdump_command_ids",
            ],
            "properties": {
                key: extension["properties"][key]
                for key in (
                    "architecture",
                    "binary_inspection",
                    "cuobjdump_command_ids",
                )
            },
            "allOf": extension["allOf"],
        }
        validator = jsonschema.Draft202012Validator(fragment)
        uninspected = {
            "architecture": {
                "derived_from": (
                    "nvidia_smi_and_torch_cuda_capability_agreement"
                ),
                "compute_capability": None,
                "equivalent_torch_cuda_arch_list": None,
                "compiled_sass_targets": [],
                "compiled_ptx_targets": [],
            },
            "binary_inspection": {
                "status": "NOT_RUN",
                "reason": "binary unavailable",
                "command_ids": [],
                "evidence_file_ids": [],
                "sass_target_present": None,
                "ptx_target_present": None,
            },
            "cuobjdump_command_ids": [],
        }
        self.assertFalse(list(validator.iter_errors(uninspected)))

        fabricated_target = copy.deepcopy(uninspected)
        fabricated_target["architecture"]["compiled_sass_targets"] = [
            "sm_120"
        ]
        self.assertTrue(list(validator.iter_errors(fabricated_target)))

        fabricated_equivalence = copy.deepcopy(uninspected)
        fabricated_equivalence["architecture"][
            "equivalent_torch_cuda_arch_list"
        ] = "12.0+PTX"
        self.assertTrue(list(validator.iter_errors(fabricated_equivalence)))

    def test_sanitizer_summary_parsers_fail_closed(self) -> None:
        memcheck = (
            "========= LEAK SUMMARY: 0 bytes leaked in 0 allocations\n"
            "========= ERROR SUMMARY: 0 errors\n"
        )
        racecheck = (
            "========= RACECHECK SUMMARY: 0 hazards displayed "
            "(0 errors, 0 warnings)\n"
        )
        self.assertEqual(
            run_preflight.sanitizer_error_count("memcheck", memcheck), 0
        )
        self.assertEqual(
            run_preflight.sanitizer_error_count("racecheck", racecheck), 0
        )
        self.assertEqual(
            run_preflight.sanitizer_error_count(
                "initcheck", "========= ERROR SUMMARY: 0 errors\n"
            ),
            0,
        )
        self.assertIsNone(
            run_preflight.sanitizer_error_count("racecheck", "")
        )
        self.assertIsNone(
            run_preflight.sanitizer_error_count(
                "synccheck",
                "ERROR SUMMARY: 0 errors\nERROR SUMMARY: 0 errors\n",
            )
        )
        self.assertGreater(
            run_preflight.sanitizer_error_count(
                "racecheck",
                "RACECHECK SUMMARY: 1 hazards displayed "
                "(0 errors, 1 warnings)",
            )
            or 0,
            0,
        )

    def test_native_host_validation_fails_on_ambiguity(self) -> None:
        clean = run_preflight.native_host_validation_errors(
            container_detection_ok=True,
            container_detection_output="none",
            cgroup_evidence={
                "host_cgroup": "0::/init.scope\n",
                "host_self_cgroup": "0::/user.slice/test.scope\n",
            },
            present_container_markers=[],
            cgroup_container_tokens=[],
            caller_container_marker=False,
        )
        self.assertEqual(clean, [])
        missing = run_preflight.native_host_validation_errors(
            container_detection_ok=True,
            container_detection_output="none",
            cgroup_evidence={
                "host_cgroup": "",
                "host_self_cgroup": "0::/scope\n",
            },
            present_container_markers=[],
            cgroup_container_tokens=[],
            caller_container_marker=False,
        )
        self.assertTrue(any("host_cgroup" in item for item in missing))
        detected = run_preflight.native_host_validation_errors(
            container_detection_ok=False,
            container_detection_output="docker",
            cgroup_evidence={
                "host_cgroup": "0::/docker/test\n",
                "host_self_cgroup": "0::/docker/test\n",
            },
            present_container_markers=["/.dockerenv"],
            cgroup_container_tokens=["docker"],
            caller_container_marker=True,
        )
        self.assertGreaterEqual(len(detected), 5)

    def test_verification_output_must_be_byte_empty(self) -> None:
        self.assertTrue(
            run_preflight.verification_outputs_are_empty(
                command_ok=True, stdout="", stderr=""
            )
        )
        self.assertFalse(
            run_preflight.verification_outputs_are_empty(
                command_ok=True,
                stdout="??5?????? c /etc/example\n",
                stderr="",
            )
        )
        self.assertFalse(
            run_preflight.verification_outputs_are_empty(
                command_ok=False, stdout="", stderr=""
            )
        )

    def test_dpkg_ownership_and_nvdisasm_version_fail_closed(self) -> None:
        expected_output = (
            "cuda-nvdisasm-13-0: "
            "/usr/local/cuda-13.0/bin/nvdisasm\n"
        )
        self.assertTrue(
            run_preflight.dpkg_ownership_matches(
                command_ok=True,
                stdout=expected_output,
                stderr="",
                expected_package="cuda-nvdisasm-13-0",
                expected_path="/usr/local/cuda-13.0/bin/nvdisasm",
            )
        )
        self.assertFalse(
            run_preflight.dpkg_ownership_matches(
                command_ok=True,
                stdout="other-package: /usr/local/cuda-13.0/bin/nvdisasm\n",
                stderr="",
                expected_package="cuda-nvdisasm-13-0",
                expected_path="/usr/local/cuda-13.0/bin/nvdisasm",
            )
        )
        self.assertFalse(
            run_preflight.dpkg_ownership_matches(
                command_ok=False,
                stdout=expected_output,
                stderr="query failed\n",
                expected_package="cuda-nvdisasm-13-0",
                expected_path="/usr/local/cuda-13.0/bin/nvdisasm",
            )
        )
        version_output = """nvdisasm: NVIDIA (R) CUDA disassembler
Cuda compilation tools, release 13.0, V13.0.85
Build cuda_13.0.r13.0/compiler.36400806_0
"""
        self.assertEqual(
            run_preflight.extract_version("nvdisasm", version_output),
            "13.0.85",
        )

    def test_platform_lock_exact_match_and_mismatch(self) -> None:
        system_lock = json.loads(
            (ROOT / "preflight" / "system-packages.lock.json").read_text()
        )
        observed = {
            **system_lock["platform"],
            **system_lock["scope"],
            "cuda_home": system_lock["environment"]["cuda_home"],
            "python_environment": system_lock["environment"][
                "python_environment"
            ],
            "python_requirements_lock": system_lock["environment"][
                "python_requirements_lock"
            ],
        }
        self.assertEqual(
            run_preflight.verify_platform_lock(system_lock, observed)["status"],
            "PASS",
        )
        mismatch = dict(observed)
        mismatch["python_abi"] = "cp313"
        result = run_preflight.verify_platform_lock(system_lock, mismatch)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("python_abi" in item for item in result["errors"]))

    def test_dependency_lock_rejects_extra_distribution(self) -> None:
        locked = run_preflight.parse_requirements_lock(
            ROOT / "preflight" / "requirements-e00.txt"
        )
        installed = [
            {"name": name, "version": version}
            for name, version in locked.items()
        ]
        minimal_system_lock = {"dpkg_packages": [], "tools": []}
        self.assertEqual(
            run_preflight.verify_dependency_locks(
                minimal_system_lock, "", installed
            )["status"],
            "PASS",
        )
        installed.append({"name": "unexpected-extra", "version": "1.0"})
        result = run_preflight.verify_dependency_locks(
            minimal_system_lock, "", installed
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any("unexpected-extra" in item for item in result["errors"])
        )

    def test_system_lock_tool_names_and_reported_versions(self) -> None:
        system_lock = json.loads(
            (ROOT / "preflight" / "system-packages.lock.json").read_text()
        )
        package_names = {
            item["name"] for item in system_lock["dpkg_packages"]
        }
        tools = {item["name"]: item for item in system_lock["tools"]}
        self.assertEqual(len(tools), len(system_lock["tools"]))
        self.assertIn("dash", package_names)
        self.assertEqual(tools["sh"]["invocation_path"], "/bin/sh")
        self.assertEqual(tools["sh"]["resolved_path"], "/usr/bin/dash")
        for tool in tools.values():
            self.assertIn(tool["dpkg_package"], package_names)
        for name in (
            "nvcc",
            "compute-sanitizer-real",
            "cuobjdump",
            "nvdisasm",
            "ncu",
            "nsys",
            "c++",
            "ninja",
            "python",
        ):
            self.assertRegex(tools[name]["reported_version"], r"^[^\s]+$")
        self.assertEqual(
            next(
                item
                for item in system_lock["dpkg_packages"]
                if item["name"] == "cuda-nvdisasm-13-0"
            ),
            {
                "name": "cuda-nvdisasm-13-0",
                "version": "13.0.85-1",
                "architecture": "amd64",
            },
        )
        self.assertEqual(
            tools["nvdisasm"],
            {
                "name": "nvdisasm",
                "invocation_path": "/usr/local/cuda-13.0/bin/nvdisasm",
                "resolved_path": "/usr/local/cuda-13.0/bin/nvdisasm",
                "dpkg_package": "cuda-nvdisasm-13-0",
                "version": "13.0.85",
                "reported_version": "13.0.85",
                "sha256": "3c27bded09bd877807207b62db8186a0a9a359d10311ab6e2c885f9b418c9f41",
            },
        )

    def test_git_contract_authentication_rejects_special_flags(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            path = directory / "contract.txt"
            path.write_bytes(b"contract\n")
            object_id = run_preflight.git_blob_sha1(path)
            staged = f"100644 {object_id} 0\tcontract.txt\0".encode()
            with mock.patch.object(run_preflight, "ROOT", directory):
                passing = run_preflight.verify_contract_git_state(
                    staged_output=staged,
                    flags_output=b"H contract.txt\0",
                    paths=("contract.txt",),
                )
                flagged = run_preflight.verify_contract_git_state(
                    staged_output=staged,
                    flags_output=b"h contract.txt\0",
                    paths=("contract.txt",),
                )
            self.assertEqual(passing["status"], "PASS")
            self.assertEqual(flagged["status"], "FAIL")

    def test_audit_checkpoint_ready_release_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            ready = directory / "ready.json"
            release = directory / "release"
            errors: list[BaseException] = []

            def worker() -> None:
                try:
                    audit_checkpoint(
                        ready_file=str(ready),
                        release_file=str(release),
                        timeout_seconds=5.0,
                    )
                except BaseException as error:
                    errors.append(error)

            thread = threading.Thread(target=worker)
            thread.start()
            deadline = time.monotonic() + 3.0
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(ready.is_file())
            payload = json.loads(ready.read_text())
            self.assertEqual(payload["protocol"], "e00-process-audit-v1")
            self.assertEqual(payload["pid"], os.getpid())
            release.write_bytes(b"release\n")
            thread.join(timeout=3.0)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])

    def test_malformed_zero_exit_process_snapshot_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            stage = Path(raw_directory)
            (stage / "commands").mkdir()
            stdout_path = stage / "commands" / "process_bad.stdout.txt"
            stderr_path = stage / "commands" / "process_bad.stderr.txt"
            stdout_path.write_text("{}\n")
            stderr_path.write_text("")
            command_record = {
                "id": "process_bad",
                "argv": ["unit-process-query"],
                "exit_code": 0,
                "finished_at_utc": "2026-07-22T00:00:00Z",
                "stdout": {
                    "path": "commands/process_bad.stdout.txt",
                    "sha256": "0" * 64,
                    "size_bytes": stdout_path.stat().st_size,
                },
                "stderr": {
                    "path": "commands/process_bad.stderr.txt",
                    "sha256": "0" * 64,
                    "size_bytes": 0,
                },
            }
            recorder = run_preflight.EvidenceRecorder(stage, {})
            snapshot = recorder._snapshot_from_command(
                phase="before",
                command_record=command_record,
            )

            self.assertEqual(snapshot["query_exit_code"], 0)
            self.assertTrue(recorder.audit_errors)
            self.assertIn(
                "invalid process snapshot process_bad",
                recorder.audit_errors[0],
            )
            self.assertEqual(snapshot["graphics_processes"], [])
            self.assertEqual(snapshot["allowed_compute_processes"], [])
            self.assertEqual(snapshot["foreign_compute_processes"], [])
            self.assertEqual(snapshot["unknown_processes"], [])
            self.assertIs(recorder.snapshots[-1], snapshot)

    def test_evidence_reference_validation_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            stage = Path(raw_directory)
            (stage / "commands").mkdir()
            stdout = stage / "commands" / "cmd.stdout.txt"
            stderr = stage / "commands" / "cmd.stderr.txt"
            stdout.write_bytes(b"ok\n")
            stderr.write_bytes(b"")
            manifest = {
                "evidence": {
                    "files": run_preflight.enumerate_evidence_files(stage),
                    "commands": [
                        {
                            "id": "cmd",
                            "stdout": run_preflight.output_ref(
                                stage, "commands/cmd.stdout.txt"
                            ),
                            "stderr": run_preflight.output_ref(
                                stage, "commands/cmd.stderr.txt"
                            ),
                        }
                    ],
                },
                "result": {
                    "command_ids": ["cmd"],
                    "evidence_file_ids": [
                        run_preflight.artifact_id(
                            "commands/cmd.stdout.txt"
                        )
                    ],
                },
            }
            self.assertEqual(
                run_preflight.evidence_reference_errors(stage, manifest), []
            )
            tampered = copy.deepcopy(manifest)
            tampered["result"]["command_ids"] = ["missing"]
            self.assertTrue(
                run_preflight.evidence_reference_errors(stage, tampered)
            )
            stdout.write_bytes(b"changed\n")
            self.assertTrue(
                run_preflight.evidence_reference_errors(stage, manifest)
            )

    def test_rename_noreplace_preserves_existing_target(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            source = directory / "source"
            target = directory / "target"
            source.mkdir()
            target.mkdir()
            with self.assertRaises(OSError):
                run_preflight.rename_noreplace(source, target)
            self.assertTrue(source.is_dir())
            self.assertTrue(target.is_dir())

    def test_terminate_process_group_kills_sigterm_ignoring_root(self) -> None:
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import signal,time;"
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
                    "time.sleep(30)"
                ),
            ],
            start_new_session=True,
        )
        try:
            time.sleep(0.1)
            self.assertTrue(
                run_preflight.terminate_process_group(
                    process, grace_seconds=0.1
                )
            )
            process.wait(timeout=2.0)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, 9)
                process.wait(timeout=2.0)

    def _fake_capture(
        self,
        recorder: run_preflight.EvidenceRecorder,
        *,
        target_command_id: str,
        phase: str,
        supervised_root: dict[str, object] | None,
        environment: dict[str, str],
    ) -> dict[str, object]:
        allowed: list[dict[str, object]] = []
        if phase == "during":
            payload = json.loads(
                (
                    recorder.stage
                    / "audit"
                    / f"{target_command_id}.ready.json"
                ).read_text()
            )
            allowed = [
                {
                    "gpu_uuid": environment["CUDA_VISIBLE_DEVICES"],
                    "pid": payload["pid"],
                    "process_start_time_ticks": payload[
                        "process_start_time_ticks"
                    ],
                }
            ]
        snapshot: dict[str, object] = {
            "phase": phase,
            "command_id": f"process_{target_command_id}_{phase}",
            "captured_at_utc": run_preflight.utc_now(),
            "query_argv": ["unit-fake-query"],
            "query_exit_code": 0,
            "raw_stdout": {},
            "raw_stderr": {},
            "graphics_processes": [],
            "allowed_compute_processes": allowed,
            "foreign_compute_processes": [],
            "unknown_processes": [],
        }
        recorder.snapshots.append(snapshot)
        return snapshot

    def test_supervised_audit_success_and_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            stage = directory / "stage"
            stage.mkdir()
            child = directory / "child.py"
            child.write_text(
                "import argparse,json,sys\n"
                f"sys.path.insert(0, {str(ROOT)!r})\n"
                "from preflight.audit_checkpoint import audit_checkpoint\n"
                "p=argparse.ArgumentParser()\n"
                "p.add_argument('--audit-ready-file')\n"
                "p.add_argument('--audit-release-file')\n"
                "p.add_argument('--audit-timeout-seconds',type=float)\n"
                "a=p.parse_args()\n"
                "audit_checkpoint(ready_file=a.audit_ready_file,"
                "release_file=a.audit_release_file,"
                "timeout_seconds=a.audit_timeout_seconds)\n"
                "print(json.dumps({'status':'pass'}))\n"
            )
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = "GPU-unit"
            recorder = run_preflight.EvidenceRecorder(stage, environment)
            with mock.patch.object(
                recorder,
                "capture_snapshot",
                side_effect=lambda **kwargs: self._fake_capture(
                    recorder, **kwargs
                ),
            ):
                recorder.run_supervised(
                    "unit_child",
                    "unit handshake",
                    [sys.executable, str(child)],
                    environment=environment,
                    timeout_seconds=10.0,
                    audit_ready_timeout_seconds=3.0,
                )
            self.assertTrue(recorder.command_ok("unit_child"))
            self.assertEqual(recorder.audit_errors, [])
            outcome = json.loads(
                (stage / "audit" / "unit_child.outcome.json").read_text()
            )
            self.assertTrue(outcome["checkpoint_verified"])
            self.assertTrue(outcome["release_published_by_collector"])
            self.assertTrue(outcome["process_group_drained"])

    def test_supervised_audit_rejects_child_release(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            stage = directory / "stage"
            stage.mkdir()
            child = directory / "premature.py"
            child.write_text(
                "import argparse,sys,time\n"
                "from pathlib import Path\n"
                f"sys.path.insert(0, {str(ROOT)!r})\n"
                "from preflight.audit_checkpoint import audit_checkpoint\n"
                "p=argparse.ArgumentParser()\n"
                "p.add_argument('--audit-ready-file')\n"
                "p.add_argument('--audit-release-file')\n"
                "p.add_argument('--audit-timeout-seconds',type=float)\n"
                "a=p.parse_args()\n"
                "Path(a.audit_release_file).write_bytes(b'release\\n')\n"
                "audit_checkpoint(ready_file=a.audit_ready_file,"
                "release_file=a.audit_release_file,"
                "timeout_seconds=a.audit_timeout_seconds)\n"
                "time.sleep(10)\n"
            )
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = "GPU-unit"
            recorder = run_preflight.EvidenceRecorder(stage, environment)
            with mock.patch.object(
                recorder,
                "capture_snapshot",
                side_effect=lambda **kwargs: self._fake_capture(
                    recorder, **kwargs
                ),
            ):
                recorder.run_supervised(
                    "premature",
                    "reject child-created release",
                    [sys.executable, str(child)],
                    environment=environment,
                    timeout_seconds=10.0,
                    audit_ready_timeout_seconds=3.0,
                )
            outcome = json.loads(
                (stage / "audit" / "premature.outcome.json").read_text()
            )
            self.assertTrue(outcome["premature_release_detected"])
            self.assertFalse(outcome["checkpoint_verified"])
            self.assertFalse(outcome["release_published_by_collector"])
            self.assertTrue(recorder.audit_errors)


if __name__ == "__main__":
    unittest.main()
