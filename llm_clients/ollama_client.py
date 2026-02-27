import base64
import aiohttp

async def query_ollama(
    prompt: str, 
    model: str, 
    image_path: str = None, 
    keep_alive: int = 0,
    base_url: str = "http://localhost:11434"
) -> str:
    """
    Asynchronously queries an Ollama Vision Language Model (VLM).
    """
    url = f"{base_url}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": f"{keep_alive}m" # Ollama accepts '0m', '5m', etc.
    }
    
    # Safely handle the image if provided
    if image_path:
        try:
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode('utf-8')
                payload["images"] = [base64_image]
        except Exception as e:
            print(f"❌ Error encoding image for Ollama ({image_path}): {e}")
            return ""

    # Execute the async request
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("response", "").strip()
                else:
                    error_text = await response.text()
                    print(f"⚠️ Ollama API Error {response.status}: {error_text}")
                    return ""
    except aiohttp.ClientConnectorError:
        print(f"❌ Failed to connect to Ollama. Is it running at {base_url}?")
        return ""
    except Exception as e:
        print(f"❌ Unexpected error querying Ollama: {e}")
        return ""