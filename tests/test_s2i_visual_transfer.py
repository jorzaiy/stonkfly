import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from experiments.a_share.run_s2i_visual_transfer_assay import frames
def test_all_preregistered_frames_are_rgb_uint8_contiguous_and_deterministic():
 a=frames(); b=frames(); assert len(a)==len(b)
 for (na,fa),(nb,fb) in zip(a,b):
  assert na==nb and fa.shape==fb.shape and fa.ndim==3 and fa.shape[2]==3 and fa.dtype==np.uint8 and fa.flags.c_contiguous and np.array_equal(fa,fb)
def test_pattern_means_are_recorded_constructively():
 for name,f in frames():
  assert np.isfinite(f.mean()) and np.all((f>=0)&(f<=255))
def test_mean_half_patterns_do_not_uint8_wrap():
 for name,f in frames():
  if any(k in name for k in ('horizontal_mean_0.5','vertical_mean_0.5','checker_2_mean_0.5','checker_8_mean_0.5')):
   assert abs(float(f.mean())/255.0-0.5) <= 1/255
