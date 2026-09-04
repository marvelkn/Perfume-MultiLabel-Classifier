import numpy as np
import pytest
from src.featurize import FeatureSpec, features_and_metadata, canonical_smiles, fingerprint_matrix, MoleculeError
from api_light import fingerprint_result, SPEC

def test_api_training_features_and_metadata_identical():
    X, metadata = features_and_metadata("OCC", SPEC)
    response = fingerprint_result("CCO")
    np.testing.assert_array_equal(X, response["fingerprint"])
    assert X.dtype == np.float32 and X.shape == (2053,)
    assert set(X[:2048]) <= {0,1}
    assert response["molecular_formula"] == "C2H6O"
    assert abs(response["molecular_weight"]-46.069) < .001

@pytest.mark.parametrize("smiles,code",[("", "INVALID_SMILES"),("bad!", "INVALID_SMILES"),("CCO.O","UNSUPPORTED_MIXTURE")])
def test_invalid_inputs(smiles,code):
    with pytest.raises(MoleculeError) as exc:
        features_and_metadata(smiles)
    assert exc.value.code == code
    assert canonical_smiles(smiles) is None

def test_stereoisomers_and_schema_identity():
    left,right = "CC1=CC[C@@H](CC1=O)C(=C)C","CC1=CC[C@H](CC1=O)C(=C)C"
    assert canonical_smiles(left) != canonical_smiles(right)
    old,new = FeatureSpec(),FeatureSpec(include_chirality=True)
    assert old.schema_id != new.schema_id
    np.testing.assert_array_equal(features_and_metadata(left,old)[0],features_and_metadata(right,old)[0])
    assert not np.array_equal(features_and_metadata(left,new)[0],features_and_metadata(right,new)[0])

def test_invalid_row_mask():
    X,mask = fingerprint_matrix(["CCO","bad!","c1ccccc1"])
    assert X.shape == (2,2053) and mask.tolist() == [True,False,True]
