from __future__ import annotations
import importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np,pytest
ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('o0',ROOT/'experiments/a_share/run_o0_olfactory_pn_inventory.py');O0=importlib.util.module_from_spec(SPEC);assert SPEC.loader;SPEC.loader.exec_module(O0)

def test_fixed_inputs_and_inventory_rules():
    O0.fixed_inputs();r,raw=O0.inventory()
    assert all(r['gates'].values()),r['gates']
    assert r['sets']['orn']['count']==2635
    assert r['sets']['orn_direct_pn']['count']==594
    assert r['sets']['eligible_excitatory_pn']['count']==290
    assert r['sets']['eligible_excitatory_pn']['node_index_sha256']=='8f9499f6d3de64d443d33fc0887e58a2cd99fb6ac962603715b2fcb79c646a65'
    assert r['sets']['eligible_excitatory_pn']['source_id_sha256']=='fe444543ee341632a0518070388565d134c8d3343ffdee541aff7c424186fe87'
    assert np.all(raw['eligible_orn_incoming_edge_count']>0)
    assert np.all(raw['eligible_kc_outgoing_edge_count']>0)

def test_panel_is_output_independent_hash_ranking():
    a,_=O0.inventory();b,_=O0.inventory();assert a['panel']==b['panel']
    assert len(a['panel']['nodes'])==8
    digests=[x['individual_digest'] for x in a['panel']['nodes']]
    assert digests==sorted(digests)
    assert all(x['neurotransmitter']=='acetylcholine' for x in a['panel']['nodes'])

def test_edge_ledgers_and_kc_coverage():
    r,_=O0.inventory();e=r['edges']['eligible_pn_to_kc']
    assert e['count']==21693 and e['positive_count']==21693 and e['negative_count']==0
    assert r['sets']['kc_covered']['count']==3795
    assert r['graph']=={'nodes':166700,'edges':25582938,'contacts':124177617,'ids_aligned':True}

def test_cli_complete_artifacts(tmp_path):
    out=tmp_path/'o0';p=subprocess.run([sys.executable,str(ROOT/'experiments/a_share/run_o0_olfactory_pn_inventory.py'),'--out',str(out)],cwd=ROOT,text=True,capture_output=True,check=True)
    assert 'O0_PN_PROXY_INVENTORY_QUALIFIED' in p.stdout
    assert {x.name for x in out.iterdir()}=={'o0_raw.npz','o0_formal.json','o0_report.md','o0_manifest.json'}
    f=json.loads((out/'o0_formal.json').read_text());m=json.loads((out/'o0_manifest.json').read_text())
    assert f['verdict']=='O0_PN_PROXY_INVENTORY_QUALIFIED' and m['hashes_start']==m['hashes_end']

def test_output_protection_and_failure_chain(tmp_path):
    p=tmp_path/'occupied';p.mkdir();(p/'keep').write_text('x')
    with pytest.raises(RuntimeError,match='refusing to overwrite'):O0.prepare(p)
    fdir=tmp_path/'failure';f=O0.failure(fdir,RuntimeError('boom'));assert f['verdict']=='O0_IMPLEMENTATION_FAILURE'
    assert all((fdir/x).is_file() for x in ('o0_raw.npz','o0_formal.json','o0_report.md','o0_manifest.json'))
