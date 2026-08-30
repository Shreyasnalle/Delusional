import gradio as gr
from main import app

demo = gr.Blocks(title="GenAI Fraud Defense Backend API")
with demo:
    gr.Markdown("# 🛡️ GenAI Fraud Defense Backend API Server is Live")
    gr.Markdown("FastAPI Endpoints are active and listening.")

app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
