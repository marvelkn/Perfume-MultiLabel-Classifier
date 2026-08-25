"""
app.py — Entry point untuk Hugging Face Spaces (Gradio SDK)

Mengintegrasikan FastAPI (api_light.py) dengan Gradio UI minimal
agar bisa di-hosting di HF Spaces Gradio SDK (GRATIS, 16GB RAM).

Endpoint /fingerprint tetap bisa diakses dari mobile app seperti biasa.
"""
import gradio as gr
from api_light import app as fastapi_app

# ── Minimal Gradio UI ─────────────────────────────────────────
with gr.Blocks(title="Essenza Fingerprint API 🧪", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧪 Essenza Fingerprint API
    
    Backend API untuk aplikasi mobile **Essenza** — sistem prediksi profil aroma parfum.
    
    ---
    
    ## Endpoint Utama
    
    **`POST /fingerprint`** — Konversi SMILES → Morgan Fingerprint [2053 fitur]
    
    ```bash
    curl -X POST {base_url}/fingerprint \\
      -H "Content-Type: application/json" \\
      -d '{"smiles": "O=Cc1ccc(O)c(OC)c1"}'
    ```
    
    ---
    
    ## Cara Kerja
    
    1. Mobile app mengirim struktur kimia (SMILES) ke endpoint ini
    2. RDKit menghitung **Morgan Fingerprint** (2048-bit, radius=2) + 5 physical descriptors
    3. Total **2053 fitur** dikembalikan ke mobile app
    4. Mobile app menjalankan **25 model XGBoost ONNX** secara on-device untuk prediksi aroma
    
    *(Inferensi model dilakukan 100% di HP — bukan di server ini)*
    """)
    
    with gr.Row():
        smiles_input = gr.Textbox(
            label="Coba di sini (SMILES)",
            placeholder="Contoh: O=Cc1ccc(O)c(OC)c1  (Vanillin)",
            scale=3
        )
        test_btn = gr.Button("Test API", variant="primary", scale=1)
    
    output = gr.JSON(label="Response /fingerprint")
    
    def test_fingerprint(smiles: str):
        if not smiles.strip():
            return {"error": "Masukkan SMILES terlebih dahulu"}
        import requests
        try:
            # Call our own /fingerprint endpoint
            from api_light import compute_fingerprint, FingerprintRequest
            from fastapi.testclient import TestClient
            client = TestClient(fastapi_app)
            resp = client.post("/fingerprint", json={"smiles": smiles.strip()})
            data = resp.json()
            if resp.status_code == 200:
                # Only show first 10 bits for readability
                fp = data.get("fingerprint", [])
                data["fingerprint_preview"] = fp[:10]
                data["fingerprint"] = f"[{len(fp)} values — hidden for display]"
            return data
        except Exception as e:
            return {"error": str(e)}
    
    test_btn.click(fn=test_fingerprint, inputs=smiles_input, outputs=output)
    smiles_input.submit(fn=test_fingerprint, inputs=smiles_input, outputs=output)

# ── Mount Gradio ke FastAPI ───────────────────────────────────
# Endpoint /fingerprint tetap accessible, Gradio UI ada di /ui
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")
