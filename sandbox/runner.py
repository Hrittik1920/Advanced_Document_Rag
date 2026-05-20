#sandbox/runner.py
import io
import json
import math
import sys
import traceback
from contextlib import redirect_stdout
from fastapi import FastAPI
from pydantic import BaseModel
SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
    "range": range, "round": round, "int": int, "float": float, "str": str,
    "print": print, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    # Added basic types often needed by LLMs:
    "list": list, "dict": dict, "set": set, "tuple": tuple, "bool": bool, 
}

def get_safe_globals():
    """Builds the global environment safely, catching missing imports."""
    env = {
        "__builtins__": SAFE_BUILTINS,
        "math": math,
    }
    
    # Safely attempt to load heavy libraries
    try:
        import numpy as np
        env["numpy"] = np
        env["np"] = np          # alias — LLMs often use `np` shorthand
    except ImportError:
        pass  # Skip if not installed, or log the error

    try:
        import pandas as pd
        env["pandas"] = pd
        env["pd"] = pd          # alias
    except ImportError:
        pass
        
    return env
#using fast api so the model to talk to the isolated sandbox

app = FastAPI()

class CodePayload(BaseModel):
    code: str
    
@app.post("/")
async def connection():
    return{"sandbox":"is reachable"}

@app.post("/run")
async def execute_code(payload: CodePayload):
    if not payload.code:
        return {"ok": False, "error": "No code provided"}

    out = io.StringIO()
    safe_env = get_safe_globals()
    
    try:
        with redirect_stdout(out):
            exec(compile(payload.code, "<sandbox>", "exec"), safe_env, {})
        return {"ok": True, "stdout": out.getvalue()}
    except Exception:
        return {
            "ok": False,
            "stdout": out.getvalue(),
            "error": traceback.format_exc(),
        }

def main():
    payload = json.loads(sys.stdin.read() or "{}")
    code = payload.get("code", "")

    if not code:
        print(json.dumps({"ok": False, "error": "No code provided"}))
        return

    out = io.StringIO()
    safe_env = get_safe_globals()
    
    try:
        with redirect_stdout(out):
            exec(compile(code, "<sandbox>", "exec"), safe_env, {})
        print(json.dumps({"ok": True, "stdout": out.getvalue()}))
    except Exception:
        print(json.dumps({
            "ok": False,
            "stdout": out.getvalue(),
            "error": traceback.format_exc(),
        }))

if __name__ == "__main__":
    main()