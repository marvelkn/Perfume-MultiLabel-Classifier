"""Gradio CPU adapter; no GPU allocation and no dependency monkey patches."""
import os
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "false")
import gradio as gr
from api_light import fingerprint_result
from src.featurize import MoleculeError

def fingerprint_fn(smiles, compound_name=""):
    try:
        return fingerprint_result(smiles,compound_name)
    except MoleculeError as exc:
        return {"status":"error","error":{"code":exc.code,"message":str(exc)}}
    except Exception:
        # Do not expose stack traces or third-party response contents.
        return {"status":"error","error":{"code":"SERVICE_ERROR","message":"The feature service could not complete the request."}}

demo = gr.Interface(fn=fingerprint_fn,inputs=[gr.Textbox(label="SMILES"),gr.Textbox(label="Compound name (optional)")],
                    outputs=gr.JSON(),title="Essenza molecular features",api_name="predict",
                    description="Single-molecule RDKit features. The mobile app runs the odor models locally.",
                    flagging_mode="never")
demo.queue(default_concurrency_limit=1,max_size=8)
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",server_port=7860)
