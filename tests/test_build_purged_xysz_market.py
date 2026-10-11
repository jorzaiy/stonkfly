import tempfile
import unittest
from pathlib import Path

from tools.build_purged_xysz_market import BuildBlocked, PLAN_SHA, datev, qualified, load_inputs, artifact_path, equity_pit_baseline

class PurgedMarketBuilderTests(unittest.TestCase):
    def test_code_and_date_normalization(self):
        self.assertEqual(qualified("600001"), "600001.SH")
        self.assertEqual(qualified("300001.SZ"), "300001.SZ")
        self.assertEqual(datev("20240102"), "2024-01-02")

    def test_declared_artifact_missing_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            mp=Path(td)/"manifest.json"; mp.write_text("{}")
            with self.assertRaises(BuildBlocked): artifact_path(mp, "missing.csv")

    def test_equity_pit_allows_post_2021_ipo_before_signal(self):
        self.assertTrue(equity_pit_baseline([{"ANN_DATE":"20210408","CHANGE_DATE":"20210601"}], "2021-07-01"))

    def test_equity_pit_rejects_announcement_after_signal(self):
        self.assertFalse(equity_pit_baseline([{"ANN_DATE":"2023-08-16","CHANGE_DATE":"2023-08-20"}], "2022-02-07"))

    def test_plan_pin_is_immutable(self):
        self.assertEqual(len(PLAN_SHA), 64)
        self.assertTrue(Path("runs/wc006_v3_2021_inputs/20260925T-purged-candidate-stream-2021-2023-v2/xysz_gap_plan.json").is_file())

if __name__ == "__main__":
    unittest.main()
