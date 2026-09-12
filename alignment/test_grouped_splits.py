"""Regression checks for transitive groups, preserved cohorts and leakage rejection."""
import copy
import json
import numpy as np
import pytest
from alignment.grouped_splits import (POLICY, structure_feature_groups, outer_split,
                                      inner_folds, validate_grouped_split)
from alignment import experiments as e


def test_transitive_structure_and_feature_identity():
    # A/B differ in stereo; B/C share deliberately colliding features; all three stay together.
    smiles = ["C[C@H](O)F", "C[C@@H](O)F", "CCO", "CCN"]
    x = np.asarray([[1,0],[0,1],[0,1],[1,1]],dtype=np.uint8)
    groups = structure_feature_groups(smiles, x)
    assert groups.tolist() == [0,0,0,1]
    # Label equality is deliberately not an input to grouping.
    assert np.array_equal(groups, structure_feature_groups(smiles,x))


def test_real_stereoisomers_stay_together():
    from alignment.perfume_data import extract_features, load_config
    import pandas as pd
    smiles = ["C[C@H](O)F","C[C@@H](O)F","CCO","CCN"]
    x,_ = extract_features(pd.DataFrame({"smiles":smiles}),load_config())
    groups = structure_feature_groups(smiles,x)
    assert groups[0] == groups[1]
    assert groups[2] != groups[3]


def fixture_split():
    groups = np.repeat(np.arange(50),2)
    y = np.tile([0,1],50)
    train,test = outer_split(groups,.2,42)
    folds = inner_folds(train,y,groups,5,42)
    split = {"train":train.tolist(),"test":test.tolist(),"folds":{"aroma":folds},
             "eligible_labels":["aroma"],"groups":groups.tolist(),"group_policy":POLICY}
    return split,groups,y


def test_preserves_all_rows_and_excludes_groups_from_all_partitions():
    split,groups,y = fixture_split()
    assert validate_grouped_split(split,groups)["folds_verified"] == 5
    assert sorted(split["train"]+split["test"]) == list(range(100))
    assert fixture_split()[0] == split
    for fold in split["folds"]["aroma"]:
        assert set(y[fold["train"]]) == set(y[fold["validation"]]) == {0,1}


@pytest.mark.parametrize("part",["outer","inner"])
def test_rejects_leakage_even_when_row_indices_do_not_overlap(part):
    split,groups,_ = fixture_split()
    if part=="outer":
        a,b=split["train"],split["test"]
    else:
        a,b=split["folds"]["aroma"][0]["train"],split["folds"]["aroma"][0]["validation"]
    a[0],b[0]=b[0],a[0]
    with pytest.raises(ValueError,match="group leakage"):
        validate_grouped_split(split,groups)


def test_infeasible_positive_group_stops_instead_of_dropping_molecules():
    groups=np.repeat(np.arange(5),4)
    y=(groups==0).astype(int)
    with pytest.raises(ValueError,match="No feasible grouped CV"):
        inner_folds(np.arange(20),y,groups,5,42)


def test_runner_rejects_old_dataset_before_loading_arrays(tmp_path,monkeypatch):
    (tmp_path/"dataset_manifest.json").write_text(json.dumps({"protocol":"perfume-five-v1"}))
    monkeypatch.setattr(e.np,"load",lambda *a,**k:pytest.fail("Should reject before loading"))
    with pytest.raises(ValueError,match="Legacy ungrouped"):
        e.preflight(tmp_path)


@pytest.mark.parametrize("operating_system", ["Windows", "Linux"])
def test_campus_runner_accepts_windows_and_linux(monkeypatch, operating_system):
    amendment = e.read(e.ROOT / "reports/campus_20260909/full/amendment.json")
    monkeypatch.setattr(e.platform, "system", lambda: operating_system)
    monkeypatch.setattr(e.platform, "node", lambda: "CAMPUS-WORKSTATION")
    assert "CAMPUS-WORKSTATION".casefold() != amendment["excluded_laptop_host"].casefold()
    e.check_host()


def test_campus_runner_still_rejects_original_laptop(monkeypatch):
    amendment = e.read(e.ROOT / "reports/campus_20260909/full/amendment.json")
    monkeypatch.setattr(e.platform, "system", lambda: "Linux")
    monkeypatch.setattr(e.platform, "node", lambda: amendment["excluded_laptop_host"])
    with pytest.raises(RuntimeError, match="original laptop"):
        e.check_host()


def test_campus_compute_profile_matches_supplied_dxdiag():
    profile = e.compute_profile()
    assert profile["cpu"] == "Intel Core i7-8700K"
    assert profile["physical_cores"] == 6
    assert profile["logical_cpus"] == 12
    assert profile["ram_gib"] == 32
    assert profile["gpu"] == "NVIDIA GeForce GTX 1080 Ti 11 GB"
    assert profile["threads_per_model"] == 6
    assert profile["execution_device"] == "cpu"
    assert profile["run_models_sequentially"] is True
    assert profile["gpu_acceleration_enabled"] is False


def test_preflight_recomputes_group_ids_from_structures(tmp_path,monkeypatch):
    # Integrity checks are bypassed in this fixture to specifically test the semantic guard.
    import pandas as pd
    from alignment.perfume_data import extract_features, load_config
    smiles=["CCO","CO","CCN","CCC"]
    cfg=load_config()
    (tmp_path/"experiment_protocol.json").write_text("{}")
    pd.DataFrame({"source_row":range(4),"smiles":smiles}).to_csv(tmp_path/"records.csv",index=False)
    meta={"group_policy":POLICY,"files":{},"labels":["x"],"summary_labels":["x"]}
    split={"group_policy":POLICY,"groups":[0,0,0,0],"eligible_labels":["x"]}
    values={"dataset_manifest.json":meta,"protocol_v3.json":{
        "id":e.RUN_ID,"dataset_manifest_sha256":"hash","environment":e.environment(),
        "group_policy":POLICY,"class_imbalance":{"policy":e.WEIGHTING_POLICY},
        "tuning":{"seconds_per_study":10,"seconds_per_algorithm":20}},
        "config.json":cfg,"splits.json":split}
    monkeypatch.setattr(e,"read",lambda p:values[p.name])
    monkeypatch.setattr(e,"digest",lambda p:"hash")
    with pytest.raises(ValueError,match="inconsistent structure"):
        e.preflight(tmp_path)


def test_collect_results_refuses_active_run(tmp_path):
    from alignment.collect_results import collect
    run=tmp_path/"runs/perfume-five-grouped-v3"
    run.mkdir(parents=True)
    (run/".alignment.lock").touch()
    with pytest.raises(RuntimeError,match="Stop/wait"):
        collect(tmp_path)


def test_collect_results_marks_partial_and_verifies_archive(tmp_path):
    from alignment.collect_results import collect
    import zipfile,hashlib
    required=["requirements_training.txt","config.yaml","WINDOWS_GUIDE.md",
              "RUN_PERFUME_FIVE.cmd","RUN_PERFUME_FIVE.sh",
              "COLLECT_CAMPUS_RESULTS.cmd","COLLECT_CAMPUS_RESULTS.sh",
              "reports/campus_20260909/full/amendment.json",
              "notebooks/02_perfume_five_preprocessing.ipynb",
              "runs/perfume-five-grouped-v3/partial.json"]
    for name in required:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text("{}")
    path=collect(tmp_path)
    with zipfile.ZipFile(path) as z:
        meta=json.loads(z.read("RESULTS_MANIFEST.json"))
        assert meta["complete"] is False
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in meta["sha256"].items())
    assert path.with_suffix(".zip.sha256").is_file()
