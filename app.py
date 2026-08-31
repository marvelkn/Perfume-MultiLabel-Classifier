"""
app.py — Hugging Face Spaces entry point.

Fixes dua bug environment HF:
1. huggingface_hub >= 0.21 menghapus HfFolder → ditambahkan stub
2. Starlette 1.x mengubah signature TemplateResponse → dicompat-patch
   (Gradio 4.x pakai old-style: TemplateResponse(name, context)
    Starlette 1.x requires new-style: TemplateResponse(request, name, context))
"""

# ── Fix 1: HfFolder stub ─────────────────────────────────────────────────────
import huggingface_hub as _hfhub
if not hasattr(_hfhub, 'HfFolder'):
    class _HfFolder:
        @classmethod
        def get_token(cls): return None
        @classmethod
        def save_token(cls, token): pass
    _hfhub.HfFolder = _HfFolder

# ── Fix 2: TemplateResponse compat shim ─────────────────────────────────────
# Gradio 4.x calls: TemplateResponse("template.html", {"request": req, ...})
# Starlette 1.x requires: TemplateResponse(req, "template.html", {...})
import starlette.templating as _starlette_tpl

_original_TR = _starlette_tpl.Jinja2Templates.TemplateResponse

def _compat_TR(self, *args, **kwargs):
    # Old-style call: first arg is a string (template name)
    if args and isinstance(args[0], str):
        name = args[0]
        context = args[1] if len(args) > 1 else kwargs.pop("context", {})
        request = context.get("request")
        # Convert to new-style: TemplateResponse(request, name, context)
        return _original_TR(self, request, name, context)
    # New-style call: first arg is a Request object
    return _original_TR(self, *args, **kwargs)

_starlette_tpl.Jinja2Templates.TemplateResponse = _compat_TR

# ── Gradio app ────────────────────────────────────────────────────────────────
import gradio as gr
from api_light import compute_fingerprint, name_to_smiles
from fastapi import HTTPException


def fingerprint_fn(smiles: str, compound_name: str) -> dict:
    """Compute Morgan Fingerprint dari SMILES atau nama senyawa."""
    smiles = (smiles or "").strip()
    compound_name = (compound_name or "").strip()

    if not smiles and not compound_name:
        return {"error": "Berikan smiles atau compound_name"}

    meta = {"iupac_name": None, "molecular_formula": None, "molecular_weight": None}
    name = compound_name or "Unknown"

    try:
        if compound_name and not smiles:
            result = name_to_smiles(compound_name)
            smiles = result["smiles"]
            name = compound_name
            meta = {k: result[k] for k in ["iupac_name", "molecular_formula", "molecular_weight"]}

        fp = compute_fingerprint(smiles)
        wt = fp[-5]
        if wt > 400:
            return {"error": f"Senyawa terlalu berat ({wt:.2f} g/mol)"}

        return {
            "smiles": smiles,
            "compound_name": name,
            "fingerprint": fp,
            "predictions": {},
            "iupac_name": meta["iupac_name"],
            "molecular_formula": meta["molecular_formula"],
            "molecular_weight": wt,
            "warning": None
        }
    except HTTPException as e:
        return {"error": e.detail}
    except Exception as e:
        return {"error": str(e)}


demo = gr.Interface(
    fn=fingerprint_fn,
    inputs=[
        gr.Textbox(label="SMILES", placeholder="Contoh: O=Cc1ccc(O)c(OC)c1"),
        gr.Textbox(label="Compound Name (opsional)", placeholder="Contoh: vanillin"),
    ],
    outputs=gr.JSON(label="Response"),
    title="🧪 Essenza Fingerprint API",
    description="API untuk komputasi Morgan Fingerprint. Mobile app call: POST /run/predict",
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
