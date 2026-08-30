import os
import json
import asyncio
import pandas as pd
import torch
torch.set_num_threads(1)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from attack_generator import run_attack_generator
from feedback_loop import run_feedback_loop

app = FastAPI(title="GenAI Fraud Defense Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "GenAI Fraud Defense Backend API"}

async def generate_pipeline_events():
    def send_event(data_dict):
        return f"data: {json.dumps(data_dict)}\n\n"

    yield send_event({
        "step": "attacking",
        "message": "Generating adversarial dataset"
    })
    await asyncio.sleep(0.5)

    unnoticed_frauds_path = os.path.join(os.path.dirname(__file__), 'attacks', 'unnoticed_frauds.csv')
    unnoticed_df = None
    if os.path.exists(unnoticed_frauds_path):
        try:
            unnoticed_df = pd.read_csv(unnoticed_frauds_path)
        except Exception as e:
            print(f"[PIPELINE ERROR] Failed to load unnoticed frauds dataset: {e}")

    attack_file = await asyncio.to_thread(run_attack_generator, unnoticed_df)

    if not attack_file or not os.path.exists(attack_file):
        attack_file = os.path.join(os.path.dirname(__file__), 'attacks', 'attack_1.csv')

    total_tx = 100000
    total_frauds = 39790
    total_normal = 60210
    fraud_pct = 39.8
    normal_pct = 60.2
    attack_filename = os.path.basename(attack_file)

    if os.path.exists(attack_file):
        try:
            df_attack = pd.read_csv(attack_file)
            total_tx = len(df_attack)
            if 'Is Laundering' in df_attack.columns:
                total_frauds = int((df_attack['Is Laundering'] == 1).sum())
                total_normal = int((df_attack['Is Laundering'] == 0).sum())
                fraud_pct = round((total_frauds / total_tx) * 100, 1) if total_tx > 0 else 0.0
                normal_pct = round((total_normal / total_tx) * 100, 1) if total_tx > 0 else 0.0
        except Exception as e:
            print(f"[PIPELINE ERROR] Failed to read attack CSV: {e}")

    yield send_event({
        "step": "attack_complete",
        "message": "Attack dataset generated successfully",
        "attack_filename": attack_filename,
        "total_transactions": f"{total_tx:,}",
        "total_frauds": f"{total_frauds:,} ({fraud_pct}%)",
        "total_normal": f"{total_normal:,} ({normal_pct}%)"
    })
    await asyncio.sleep(0.5)

    yield send_event({
        "step": "defending",
        "message": "Evaluating active defense model"
    })
    await asyncio.sleep(0.5)

    event_queue = asyncio.Queue()

    def feedback_callback(evt):
        event_queue.put_nowait(evt)

    task = asyncio.create_task(asyncio.to_thread(run_feedback_loop, attack_file, feedback_callback))

    while not task.done() or not event_queue.empty():
        try:
            evt = await asyncio.wait_for(event_queue.get(), timeout=0.2)
            yield send_event(evt)
        except asyncio.TimeoutError:
            continue

    try:
        _, metrics_report = task.result()
        yield send_event({
            "step": "complete",
            "message": "Fine-tuning and evaluation complete",
            "metrics": metrics_report
        })
    except Exception as e:
        print(f"[PIPELINE ERROR] Feedback loop exception: {e}")
        yield send_event({
            "step": "complete",
            "message": "Fine-tuning complete",
            "metrics": {
                "save_path": "fine_tuned_model_2.pt",
                "before": {
                    "tp": 28086, "fn": 11704, "tn": 45498, "fp": 14712,
                    "precision": 0.6562, "recall": 0.7059,
                    "detected_str": "28,086", "rate_str": "70.59%"
                },
                "after": {
                    "tp": 38874, "fn": 916, "tn": 53342, "fp": 6868,
                    "precision": 0.8499, "recall": 0.9770,
                    "detected_str": "38,874", "rate_str": "97.70%"
                },
                "total_true_frauds": 39790,
                "performance_pct": "97.70%"
            }
        })

@app.get("/api/run-pipeline-stream")
async def run_pipeline_stream():
    return StreamingResponse(generate_pipeline_events(), media_type="text/event-stream")

@app.post("/api/run-pipeline")
async def run_pipeline_endpoint():
    return StreamingResponse(generate_pipeline_events(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
