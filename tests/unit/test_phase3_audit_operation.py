"""CPU-only tests for the strict Phase 3 audit operation key."""

from __future__ import annotations

import copy
import dataclasses
import unittest

from kvbench.errors import SchemaValidationError
from kvbench.runtime.phase3_audit_operation import (
    PHASE3_AUDIT_OPERATION_SCHEMA_VERSION,
    Phase3AuditOperationKey,
    validate_phase3_audit_operation_set,
)
from kvbench.schema import GraphMode, RunnerKind
from kvbench.schema.phase3 import Phase3ProcessPoint
from kvbench.schema.phase3 import (
    PHASE3_FIXED_PLAN_PATH,
    PHASE3_GROWING_PLAN_PATH,
    PHASE3_PLAN_FINGERPRINTS,
)


CACHE_SHA256 = "1" * 64
MODEL_SHA256 = "2" * 64
BACKEND_SHA256 = "3" * 64
SOURCE_SHA256 = "4" * 64
EXECUTION_GIT_SHA = "5" * 40
HARDWARE_SHA256 = "6" * 64
SOFTWARE_SHA256 = "7" * 64


def fixed_point(
    *, graph_mode: GraphMode = GraphMode.EAGER
) -> Phase3ProcessPoint:
    return Phase3ProcessPoint(
        point_id=f"fixed_l-b1-l128-{graph_mode.value}-r1",
        runner_kind=RunnerKind.FIXED_L,
        graph_mode=graph_mode,
        batch_size=1,
        context_length=128,
        output_steps=1,
        process_replicate=1,
        stability_member=False,
    )


def growing_point() -> Phase3ProcessPoint:
    return Phase3ProcessPoint(
        point_id="growing_context-b4-l4096-eager-r1",
        runner_kind=RunnerKind.GROWING_CONTEXT,
        graph_mode=GraphMode.EAGER,
        batch_size=4,
        context_length=4096,
        output_steps=16,
        process_replicate=1,
        stability_member=False,
    )


def operation(
    *,
    point: Phase3ProcessPoint | None = None,
    decode_step: int = 0,
    run_id: str = "phase3-audit-operation-fixture",
) -> Phase3AuditOperationKey:
    selected_point = fixed_point() if point is None else point
    plan_path = (
        PHASE3_FIXED_PLAN_PATH
        if selected_point.runner_kind is RunnerKind.FIXED_L
        else PHASE3_GROWING_PLAN_PATH
    )
    return Phase3AuditOperationKey.from_point(
        run_id=run_id,
        point=selected_point,
        decode_step=decode_step,
        cache_layout_fingerprint=CACHE_SHA256,
        execution_git_sha=EXECUTION_GIT_SHA,
        plan_fingerprint=PHASE3_PLAN_FINGERPRINTS[plan_path],
        hardware_identity_sha256=HARDWARE_SHA256,
        software_identity_sha256=SOFTWARE_SHA256,
        model_identity_sha256=MODEL_SHA256,
        backend_identity_sha256=BACKEND_SHA256,
        source_identity_sha256=SOURCE_SHA256,
    )


class Phase3AuditOperationKeyTests(unittest.TestCase):
    def test_fixed_l_round_trip_and_exact_geometry(self) -> None:
        observed = operation()
        self.assertEqual(
            observed.schema_version,
            PHASE3_AUDIT_OPERATION_SCHEMA_VERSION,
        )
        self.assertEqual(observed.decode_step, 0)
        self.assertEqual(observed.historical_context, 128)
        self.assertEqual(observed.attended_context, 129)
        self.assertEqual(observed.capacity, 129)
        self.assertEqual(observed.dispatch_execution_mode, "eager")
        self.assertEqual(observed.allocation_execution_mode, "eager")
        self.assertEqual(
            Phase3AuditOperationKey.from_dict(observed.to_dict()),
            observed,
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            observed.decode_step = 1  # type: ignore[misc]

    def test_cuda_graph_modes_are_explicit_and_not_aliases(self) -> None:
        observed = operation(point=fixed_point(graph_mode=GraphMode.CUDA_GRAPH))
        payload = observed.to_dict()
        self.assertEqual(payload["graph_mode"], "cuda_graph")
        self.assertEqual(
            payload["dispatch_execution_mode"], "cuda_graph_replay"
        )
        self.assertEqual(payload["allocation_execution_mode"], "cuda_graph")
        self.assertEqual(Phase3AuditOperationKey.from_dict(payload), observed)

    def test_growing_context_binds_every_step_geometry(self) -> None:
        keys = tuple(
            operation(point=growing_point(), decode_step=step)
            for step in range(16)
        )
        for step, key in enumerate(keys):
            with self.subTest(step=step):
                self.assertEqual(key.historical_context, 4096 + step)
                self.assertEqual(key.attended_context, 4097 + step)
                self.assertEqual(key.capacity, 4112)
                self.assertEqual(key.dispatch_execution_mode, "eager")
                self.assertEqual(key.allocation_execution_mode, "eager")
        self.assertEqual(len({key.operation_fingerprint_sha256 for key in keys}), 16)

    def test_fingerprint_is_deterministic_and_identity_bound(self) -> None:
        first = operation()
        second = operation()
        other_run = operation(run_id="phase3-audit-operation-other-run")
        self.assertEqual(
            first.operation_fingerprint_sha256,
            second.operation_fingerprint_sha256,
        )
        self.assertNotEqual(
            first.operation_fingerprint_sha256,
            other_run.operation_fingerprint_sha256,
        )

    def test_from_point_rejects_out_of_range_and_boolean_steps(self) -> None:
        invalid_cases = (
            (fixed_point(), 1),
            (growing_point(), -1),
            (growing_point(), 16),
            (growing_point(), True),
        )
        for point, step in invalid_cases:
            with self.subTest(point=point.point_id, step=step):
                with self.assertRaises(ValueError):
                    operation(point=point, decode_step=step)

    def test_strict_parser_rejects_missing_unknown_and_wrong_types(self) -> None:
        payload = operation().to_dict()
        missing = copy.deepcopy(payload)
        del missing["source_identity_sha256"]
        with self.assertRaises(SchemaValidationError):
            Phase3AuditOperationKey.from_dict(missing)
        unknown = copy.deepcopy(payload)
        unknown["execution_mode"] = "eager"
        with self.assertRaises(SchemaValidationError):
            Phase3AuditOperationKey.from_dict(unknown)
        wrong_type = copy.deepcopy(payload)
        wrong_type["decode_step"] = False
        with self.assertRaises(SchemaValidationError):
            Phase3AuditOperationKey.from_dict(wrong_type)

    def test_parser_rejects_subsystem_aliases_in_manifest_fields(self) -> None:
        payload = operation(point=fixed_point(graph_mode=GraphMode.CUDA_GRAPH)).to_dict()
        for field, alias in (
            ("graph_mode", "cuda_graph_replay"),
            ("graph_mode", "graph"),
            ("runner_kind", "fixed"),
        ):
            tampered = copy.deepcopy(payload)
            tampered[field] = alias
            with self.subTest(field=field, alias=alias):
                with self.assertRaises(SchemaValidationError):
                    Phase3AuditOperationKey.from_dict(tampered)

    def test_parser_recomputes_modes_and_fingerprint(self) -> None:
        payload = operation(point=fixed_point(graph_mode=GraphMode.CUDA_GRAPH)).to_dict()
        mutations: tuple[tuple[str, object], ...] = (
            ("dispatch_execution_mode", "eager"),
            ("allocation_execution_mode", "eager"),
            ("operation_fingerprint_sha256", "0" * 64),
            ("point_fingerprint", "0" * 64),
            ("historical_context", 129),
            ("attended_context", 130),
            ("capacity", 130),
            ("batch_size", 4),
            ("execution_git_sha", "0" * 40),
            ("plan_fingerprint", "0" * 64),
            ("hardware_identity_sha256", "0" * 64),
            ("software_identity_sha256", "0" * 64),
        )
        for field, value in mutations:
            tampered = copy.deepcopy(payload)
            tampered[field] = value
            with self.subTest(field=field):
                with self.assertRaises(SchemaValidationError):
                    Phase3AuditOperationKey.from_dict(tampered)

    def test_non_grid_point_and_mismatched_point_fields_are_rejected(self) -> None:
        payload = operation().to_dict()
        non_grid = copy.deepcopy(payload)
        non_grid["point_id"] = "fixed_l-b1-l129-eager-r1"
        with self.assertRaises(SchemaValidationError):
            Phase3AuditOperationKey.from_dict(non_grid)
        mismatched = copy.deepcopy(payload)
        mismatched["process_replicate"] = 2
        with self.assertRaises(SchemaValidationError):
            Phase3AuditOperationKey.from_dict(mismatched)

    def test_from_point_rejects_wrong_plan_fingerprint(self) -> None:
        point = fixed_point()
        with self.assertRaises(ValueError):
            Phase3AuditOperationKey.from_point(
                run_id="phase3-audit-operation-fixture",
                point=point,
                decode_step=0,
                cache_layout_fingerprint=CACHE_SHA256,
                execution_git_sha=EXECUTION_GIT_SHA,
                plan_fingerprint="0" * 64,
                hardware_identity_sha256=HARDWARE_SHA256,
                software_identity_sha256=SOFTWARE_SHA256,
                model_identity_sha256=MODEL_SHA256,
                backend_identity_sha256=BACKEND_SHA256,
                source_identity_sha256=SOURCE_SHA256,
            )


class Phase3AuditOperationCoverageTests(unittest.TestCase):
    def test_fixed_l_requires_exactly_one_operation(self) -> None:
        key = operation()
        self.assertEqual(validate_phase3_audit_operation_set((key,)), (key,))
        with self.assertRaises(ValueError):
            validate_phase3_audit_operation_set(())
        with self.assertRaises(ValueError):
            validate_phase3_audit_operation_set((key, key))

    def test_growing_context_requires_sixteen_ordered_unique_operations(self) -> None:
        keys = tuple(
            operation(point=growing_point(), decode_step=step)
            for step in range(16)
        )
        self.assertEqual(validate_phase3_audit_operation_set(keys), keys)
        invalid_sets = (
            keys[:-1],
            tuple(reversed(keys)),
            (*keys[:8], keys[7], *keys[9:]),
        )
        for candidate in invalid_sets:
            with self.subTest(steps=tuple(key.decode_step for key in candidate)):
                with self.assertRaises(ValueError):
                    validate_phase3_audit_operation_set(candidate)

    def test_coverage_rejects_mixed_point_and_identity_bindings(self) -> None:
        keys = tuple(
            operation(point=growing_point(), decode_step=step)
            for step in range(16)
        )
        mixed_run = list(keys)
        mixed_run[8] = operation(
            point=growing_point(),
            decode_step=8,
            run_id="phase3-audit-operation-other-run",
        )
        with self.assertRaises(ValueError):
            validate_phase3_audit_operation_set(mixed_run)
        with self.assertRaises(TypeError):
            validate_phase3_audit_operation_set((keys[0], "not-a-key"))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
