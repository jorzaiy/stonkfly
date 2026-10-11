from __future__ import annotations
import importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np,pytest
ROOT=Path(__file__).resolve().parents[1];SPEC=importlib.util.spec_from_file_location('o1',ROOT/'experiments/a_share/run_o1_pn_single_event.py');O1=importlib.util.module_from_spec(SPEC);assert SPEC.loader;SPEC.loader.exec_module(O1)
from stonkfly.neural.v2_brain import HybridBrain
def graph(tmp_path):
 p=tmp_path/'g.npz';np.savez(p,ptr=np.array([0,1,1],np.int64),post=np.array([1],np.int32),weight=np.array([2.],np.float32),ids=np.array([11,12],np.int64));return p
def test_single_event_arrives_once_without_external_drive(tmp_path):
 b=HybridBrain(graph(tmp_path),dt=.1,graded_mask=np.zeros(2,np.uint8));intent=O1.inject_single_event(b,0);assert intent['count']==1 and intent['arrival_clock']==18
 b.run(2.0);assert b.g_event[1]!=0 and b.drive[0]==0 and b.drive[1]==0 and np.all(b.spike_queue_count==0)
def test_invalid_or_duplicate_intent_rejected(tmp_path):
 b=HybridBrain(graph(tmp_path),dt=.1,graded_mask=np.zeros(2,np.uint8))
 with pytest.raises(ValueError):O1.inject_single_event(b,2)
 O1.inject_single_event(b,0)
 with pytest.raises(RuntimeError):O1.inject_single_event(b,0)
def test_fixed_o0_panel_and_pins():
 _,o0=O1.fixed();assert o0['verdict']=='O0_PN_PROXY_INVENTORY_QUALIFIED';assert o0['panel']['panel_sha256']==O1.PANEL_SHA;assert len(o0['panel']['nodes'])==8
def test_cli_smoke_chain(tmp_path):
 out=tmp_path/'smoke';r=subprocess.run([sys.executable,str(ROOT/'experiments/a_share/run_o1_pn_single_event.py'),'--smoke','--out',str(out)],cwd=ROOT,text=True,capture_output=True,check=True);assert 'O1_SMOKE_OK' in r.stdout
 assert {x.name for x in out.iterdir()}=={'o1_raw.npz','o1_formal.json','o1_report.md','o1_manifest.json','launch.txt','stdout.txt'}
 f=json.loads((out/'o1_formal.json').read_text());m=json.loads((out/'o1_manifest.json').read_text());assert f['engineered_proxy'] and not f['market_used'] and f['verdict']=='O1_SMOKE_OK';assert m['hashes_start']==m['hashes_end']
def test_output_protection(tmp_path):
 p=tmp_path/'x';p.mkdir();(p/'keep').write_text('x')
 with pytest.raises(RuntimeError):O1.prepare(p)
