# -*- coding: utf-8 -*-
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict, List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_traffic_state():
    try:
        from sim import signals, currentGreen, currentYellow, vehicles, emergencyCounts, directionNumbers
        total_emergency = sum(emergencyCounts.values())
        timers = {
            "red": [max(0, s.red) for s in signals],
            "yellow": [max(0, s.yellow) for s in signals],
            "green": [max(0, s.green) for s in signals]
        }
        traffic_signals = []
        for i, signal in enumerate(signals):
            status = "GREEN" if i == currentGreen and currentYellow == 0 else "YELLOW" if i == currentGreen and currentYellow == 1 else "RED"
            traffic_signals.append({
                "id": i + 1,
                "status": status,
                "red": max(0, signal.red),
                "yellow": max(0, signal.yellow),
                "green": max(0, signal.green)
            })
        return {
            "emergency_vehicles_passed": total_emergency,
            "current_green": currentGreen + 1,
            "current_yellow": currentYellow,
            "timers": timers,
            "emergency": {
                "right": emergencyCounts["right"],
                "down": emergencyCounts["down"],
                "left": emergencyCounts["left"],
                "up": emergencyCounts["up"]
            },
            "traffic_signals": traffic_signals,
            "vehicles_crossed": {
                dir_name: vehicles[dir_name]["crossed"] 
                for dir_name in ["right", "down", "left", "up"]
            }
        }
    except ImportError:
        return {"error": "Simulation not running. Please start sim.py first."}
    except Exception as e:
        return {"error": f"Error getting traffic state: {str(e)}"}

@app.get("/")
async def root():
    return {"status": "API is running. Use /status endpoint for traffic data"}

@app.get("/status")
async def get_status() -> Dict:
    return get_traffic_state()

if __name__ == "__main__":
    host = "10.22.66.205"
    port = 8000
    print(f"Starting API server at http://{host}:{port}")
    print("To get traffic status, visit:")
    print(f"http://{host}:{port}/status")
    uvicorn.run("api:app", host=host, port=port, reload=True)