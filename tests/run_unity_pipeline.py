# examples/run_unity_pipeline.py
import json, time, threading, requests

def run():
    """
    דרישות:
      - להפעיל במקביל את api/run_adapter_http.py (שמריץ גם את WS broker)
      - פרויקט Unity עם BuildScript.PerformBuild שמייצר ארטיפקטים ל-output_dir
    """
    topic = "demo-unity-job-1"
    payload = {
        "topic": topic,
        "project_path": "/path/to/UnityProject",
        "target": "Android",
        "output_dir": "./.unity_out"
    }
    r = requests.post("http://127.0.0.1:8089/run_adapter/unity_build", json=payload, timeout=600)
    print(r.status_code, r.text)

if __name__ == "__main__":
    run()