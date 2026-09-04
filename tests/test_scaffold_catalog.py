import json
import numpy as np
import pandas as pd
import pytest
from src.splits import development_partitions
from src.export_catalog import export_catalog
from src.featurize import FeatureSpec

def test_scaffold_groups_cannot_cross_development_roles():
    groups=np.repeat(np.arange(30),4)
    Y=np.random.default_rng(1).integers(0,2,(120,3))
    splits=development_partitions(Y,groups=groups)
    assert not set(groups[splits["fit"]]) & set(groups[splits["threshold"]])
    for fold in splits["folds"]:
        a,b,c=(set(groups[fold[k]]) for k in ["train","stop","score"])
        assert not a&b and not a&c and not b&c

def test_rdkit_version_mismatch_is_explicit():
    with pytest.raises(ValueError,match="requires RDKit"):FeatureSpec(rdkit_version="2000.01.1")

def test_catalog_export_preserves_explicit_taxonomy_and_source_identity(tmp_path):
    source=tmp_path/"source.csv"
    row={"pid":"1","name":"Sample","brand":"Test","gender":"unisex","rating":"4",
         "accords":json.dumps({"catalog_specific":.7,"woody":.2})}
    pd.DataFrame([row]).to_csv(source,index=False)
    output=export_catalog(source,tmp_path/"output","https://example.org/explicit-test-fixture","fixture-v1")
    result=json.loads((output/"perfumes.json").read_text())
    assert result[0]["accords"]["catalog_specific"]==.7
    assert json.loads((output/"catalog_manifest.json").read_text())["source_sha256"]
    with pytest.raises(FileExistsError):export_catalog(source,output,"source","v1")
    row["accords"]=json.dumps({"woody":2})
    pd.DataFrame([row]).to_csv(source,index=False)
    with pytest.raises(ValueError,match="finite"):export_catalog(source,tmp_path/"invalid","source","v1")
