#sandbox/tools.py
import aiohttp
from logger_utils import log_debug

async def run_code_in_sandbox(code: str) -> str:
    """Sends code to the persistent Docker sandbox via HTTP."""
    if not code or not code.strip():
        return "Sandbox Error: No code provided."

    log_debug("[SANDBOX] Sending code to persistent sandbox API...")
    
    # Send the request to the container's exposed port
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://sandbox:9999/run", 
                json={"code": code},
                timeout=10 # Prevent infinite loops from locking the agent
            ) as response:
                output = await response.json()
    except Exception as e:
        return f"Sandbox Connection Error: {str(e)}"

    if output.get("ok"):
        stdout = output.get("stdout", "").strip()
        return stdout or "Code executed successfully with no printed output."
    else:
        error = output.get("error", "Unknown error.")
        partial = output.get("stdout", "").strip()
        if partial:
            return f"Partial output:\n{partial}\n\nError:\n{error}"
        return f"Sandbox Error:\n{error}"