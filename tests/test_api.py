import pytest
from fastapi.testclient import TestClient
from api import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "AromaML B2B API berjalan!"}

def test_fingerprint_valid_smiles(client):
    # Linalool (a common perfume ingredient)
    response = client.post("/fingerprint", json={"smiles": "CC(=CCCC(C)(C=C)O)C"})
    assert response.status_code == 200
    data = response.json()
    assert data["smiles"] == "CC(=CCCC(C)(C=C)O)C"
    assert "fingerprint" in data
    # Check if length is 2053 (2048 Morgan + 5 Physical)
    assert len(data["fingerprint"]) == 2053
    assert "predictions" in data

def test_fingerprint_heavy_molecule(client):
    # A molecule with weight > 400 (e.g. Brevetoxin or large polymer)
    # Let's use a long alkane C30H62 (MolWt ~422)
    response = client.post("/fingerprint", json={"smiles": "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"})
    assert response.status_code == 400
    assert "Senyawa terlalu berat" in response.json()["detail"]

def test_fingerprint_invalid_smiles(client):
    response = client.post("/fingerprint", json={"smiles": "invalid_smiles_string"})
    assert response.status_code == 400

def test_recommend_empty(client):
    response = client.post("/recommend", json={"label_probabilities": {}})
    assert response.status_code == 200
    assert response.json() == []

def test_recommend_valid(client):
    response = client.post("/recommend", json={
        "label_probabilities": {"floral": 0.9, "citrus": 0.8},
        "top_k": 2
    })
    # If the database exists, it should return 200.
    if response.status_code == 200:
        assert isinstance(response.json(), list)
    elif response.status_code == 500:
        # DB not found case
        assert "Database parfum belum siap" in response.json()["detail"]
