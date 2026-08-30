import spaces
import gradio as gr
from main import app

@spaces.GPU(duration=60)
def init_zero_gpu():
    return "ZeroGPU Active"

# Call once on startup so ZeroGPU registers the GPU allocation
try:
    init_zero_gpu()
except Exception as e:
    print(f"ZeroGPU Startup notice: {e}")

demo = gr.Blocks(title="GenAI Fraud Defense Backend API")
with demo:
    gr.Markdown("# 🛡️ GenAI Fraud Defense Backend API Server is Live")

demo = gr.mount_gradio_app(app, demo, path="/ui")
