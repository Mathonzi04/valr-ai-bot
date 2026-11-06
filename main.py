from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import requests

app = FastAPI()
model = joblib.load("ai_price_action_model.pkl")

class Candle(BaseModel):
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float

class Payload(BaseModel):
    symbol: str
    candles: List[Candle]

@app.post("/predict")
def predict(payload: Payload):
    try:
        df = pd.DataFrame([c.dict() for c in payload.candles])
        
        # 1. Calculate Features (matching your Colab training)
        df['vol_sma'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_sma']
        
        fvg_bull = 1 if (df['low'].iloc[-1] > df['high'].iloc[-3]) else 0
        fvg_bear = 1 if (df['high'].iloc[-1] < df['low'].iloc[-3]) else 0
        
        features = pd.DataFrame([{
            'vol_ratio': df['vol_ratio'].iloc[-1],
            'fvg_bull': fvg_bull,
            'fvg_bear': fvg_bear
        }])
        
        # 2. Get AI Prediction
        confidence = float(model.predict_proba(features)[0][1])
        
        # 3. Rule-Based Trend Filter (Fetch 1h data from Binance)
        sym = payload.symbol.replace("ZAR", "USDT")
        h1_url = f"https://api.binance.com/api/v3/klines?symbol={sym}&interval=1h&limit=50"
        h1_res = requests.get(h1_url).json()
        h1_closes = [float(c[4]) for c in h1_res]
        ema_50_1h = pd.Series(h1_closes).ewm(span=50, adjust=False).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        trend_up = 1 if current_price > ema_50_1h else 0
        
        # 4. Final Signal Logic
        signal = "HOLD"
        if confidence > 0.75 and trend_up == 1 and fvg_bull == 1:
            signal = "BUY"
            
        return {
            "signal": signal,
            "confidence": round(confidence, 4),
            "fvg_detected": fvg_bull == 1,
            "trend_up": trend_up == 1,
            "fvg_top": float(df['low'].iloc[-1]) if fvg_bull else 0,
            "fvg_bottom": float(df['high'].iloc[-3]) if fvg_bull else 0
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def health():
    return {"status": "online", "message": "AI Bot is running"}
