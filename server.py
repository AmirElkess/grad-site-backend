from email import message
from fastapi import FastAPI, WebSocket
from starlette.responses import FileResponse 
from functions.predict import predict

app = FastAPI()

@app.get("/")
async def home():
    return FileResponse('./frontend/index.html')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        message = await websocket.receive_text()
        message = predict(message)
        await websocket.send_text(message)

@app.get("/pred")
async def pred():
    predictions = predict()
    return {
        "message": "Hello World",
        "predictions": predictions
        }
