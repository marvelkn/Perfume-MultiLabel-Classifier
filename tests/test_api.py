from fastapi.testclient import TestClient
import pytest
import requests
import api_light
from app import fingerprint_fn, demo

client = TestClient(api_light.app)

def test_rest_schema_and_gradio_adapter():
    result = client.post("/fingerprint",json={"smiles":"CCO"})
    assert result.status_code == 200
    assert result.json() == fingerprint_fn("CCO")
    assert client.get("/").json()["feature_schema_id"] == result.json()["feature_schema_id"]
    assert demo.config["api_prefix"] == "/gradio_api"
    assert any(d["api_name"] == "predict" for d in demo.config["dependencies"])

@pytest.mark.parametrize("smiles,code",[("invalid!","INVALID_SMILES"),("CC.O","UNSUPPORTED_MIXTURE"),("","INVALID_SMILES")])
def test_structured_errors(smiles,code):
    result = client.post("/fingerprint",json={"smiles":smiles})
    assert result.status_code == 422 and result.json()["detail"]["code"] == code
    assert fingerprint_fn(smiles)["error"]["code"] == code

def test_version_mismatch_rejected():
    result=client.post("/fingerprint",json={"smiles":"CC","feature_schema_id":"wrong"})
    assert result.status_code == 422
    assert result.json()["detail"]["code"] == "FEATURE_SCHEMA_MISMATCH"

def test_molecular_weight_is_not_volatility_rule():
    result=client.post("/fingerprint",json={"smiles":"C"*30})
    assert result.status_code == 200
    assert result.json()["molecular_weight"] > 400

def test_smiles_does_not_require_pubchem(monkeypatch):
    def unavailable(*args,**kwargs): raise requests.Timeout()
    monkeypatch.setattr(api_light.requests,"get",unavailable)
    assert client.post("/fingerprint",json={"smiles":"CCO"}).status_code == 200
    result=client.post("/fingerprint",json={"compound_name":"ethanol"})
    assert result.status_code == 503 and result.json()["detail"]["code"] == "LOOKUP_UNAVAILABLE"
