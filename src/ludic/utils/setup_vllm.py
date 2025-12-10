# Rich Imports for Dashboard
from rich.console import Console
import time
import requests
import logging

logger = logging.getLogger("lora_7b_trainer")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def wait_for_server(url: str, console: Console, timeout_s: int = 600):
    """
    Waits for vLLM to be healthy. Raises TimeoutError if it takes too long.
    """
    console.print(f"⏳ Waiting for vLLM at {url} (timeout={timeout_s}s)...")
    logger.info(f"Connecting to vLLM at {url}")

    start_time = time.time()

    while True:
        # Check for timeout
        if time.time() - start_time > timeout_s:
            logger.error(f"Timeout waiting for vLLM at {url}")
            raise TimeoutError(
                f"vLLM server at {url} did not respond within {timeout_s} seconds."
            )

        try:
            if requests.get(f"{url}/health", timeout=1).status_code == 200:
                console.print("✅ vLLM Server is online.")
                logger.info("vLLM Server online")
                return
        except requests.RequestException:
            pass

        time.sleep(2)
