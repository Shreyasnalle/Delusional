import gradio as gr
from main import app

demo = gr.Blocks(title="GenAI Fraud Defense Backend API")
with demo:
    gr.Markdown("# 🛡️ GenAI Fraud Defense Backend API Server is Live")
    gr.Markdown("FastAPI Endpoints are active and listening.")

demo = gr.mount_gradio_app(app, demo, path="/ui")
