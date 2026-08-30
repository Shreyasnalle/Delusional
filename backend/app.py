import spaces
import gradio as gr
from main import app

@spaces.GPU(duration=60)
def init_zero_gpu():
    return "ZeroGPU Active"

demo = gr.Blocks(title="GenAI Fraud Defense Backend API")
with demo:
    gr.Markdown("# 🛡️ GenAI Fraud Defense Backend API Server is Live")
    status_text = gr.Textbox(label="ZeroGPU Status")
    
    # Wire the GPU function to the load event so it's statically detected by AST scanner
    demo.load(fn=init_zero_gpu, outputs=status_text)

app = gr.mount_gradio_app(app, demo, path="/")
