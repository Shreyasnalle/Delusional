import spaces
import gradio as gr
from main import app

@spaces.GPU
def init_gpu():
    return "ZeroGPU Initialized"

demo = gr.Blocks(title="GenAI Fraud Defense Backend API")
with demo:
    gr.Markdown("# 🛡️ GenAI Fraud Defense Backend API Server is Live")
    status_btn = gr.Button("Check GPU Status")
    status_out = gr.Textbox(label="Status")
    status_btn.click(fn=init_gpu, outputs=status_out)

demo = gr.mount_gradio_app(app, demo, path="/ui")
