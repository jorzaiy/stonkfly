import json
from experiments.a_share.run_s2i_calibration_grid import GRID, BASELINE, select

def test_exact_grid_and_baseline():
    assert len(GRID) == 12 and BASELINE in GRID

def test_selector_requires_no_worse_and_strict_improvement():
    base={"uniform_distinct_kc_set_count":5,"uniform_max_abs_kc_count_jump_over_total_kc":.2,"pattern_exact_kc_set_collision_pairs":4}
    cfg={(0.02,12.0):base,(0.05,6.0):{"uniform_distinct_kc_set_count":6,"uniform_max_abs_kc_count_jump_over_total_kc":.2,"pattern_exact_kc_set_collision_pairs":4}}
    out=select(cfg,base); assert out["selected"]["retina_half_saturation"]==.05
    assert select({(0.05,6.0):{"uniform_distinct_kc_set_count":4,"uniform_max_abs_kc_count_jump_over_total_kc":.2,"pattern_exact_kc_set_collision_pairs":4}},base)["status"]=="no_acceptable_calibration"

def test_selector_json_serializable():
    base={"uniform_distinct_kc_set_count":1,"uniform_max_abs_kc_count_jump_over_total_kc":1.0,"pattern_exact_kc_set_collision_pairs":1}
    json.dumps(select({(0.02,12.0):base},base),allow_nan=False)
