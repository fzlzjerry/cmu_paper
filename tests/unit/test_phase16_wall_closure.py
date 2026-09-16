import hashlib
import json
import unittest

from scripts.phase16_wall_closure import exact_wall_median


class WallClosureTests(unittest.TestCase):
    def fixture(self):
        row = dict(run_id="run", method_config_id="bf16", batch_size=1,
                   context_label=4096, measured_batches=3, process_median_ms=2.0)
        samples = [dict(completed_operations=256, failed_operations=0,
                        host_ns_per_operation=x, host_total_ns=x * 256,
                        cuda_ms_per_operation=2.0)
                   for x in (1_000_000, 3_000_000, 9_000_000)]
        result = {**row, "runner": {"timing": {"samples": samples}}}
        return row, result

    def calculate(self, row, result):
        data = json.dumps(result).encode()
        return exact_wall_median(data, digest=hashlib.sha256(data).hexdigest(), row=row)

    def test_raw_host_median_not_cuda_median(self):
        row, result = self.fixture()
        self.assertEqual(self.calculate(row, result), 3.0)

    def test_checksum_and_identity_fail_closed(self):
        row, result = self.fixture()
        with self.assertRaises(ValueError):
            exact_wall_median(json.dumps(result).encode(), digest="0" * 64, row=row)
        result["batch_size"] = 2
        with self.assertRaises(ValueError):
            self.calculate(row, result)

    def test_missing_invalid_and_incomplete_samples_rejected(self):
        for key, value in (("completed_operations", 128), ("failed_operations", 1),
                           ("host_ns_per_operation", float("nan")), ("host_total_ns", 1)):
            row, result = self.fixture()
            result["runner"]["timing"]["samples"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.calculate(row, result)
        row, result = self.fixture()
        result["runner"]["timing"]["samples"].pop()
        with self.assertRaises(ValueError):
            self.calculate(row, result)

    def test_device_index_mismatch_rejected(self):
        row, result = self.fixture()
        row["process_median_ms"] = 4.0
        with self.assertRaises(ValueError):
            self.calculate(row, result)
