# test_concurrency.py
import asyncio
import time
from src.models.clip import get_clip_model

async def encode_test(n):
    clip = get_clip_model("ViT-B-32__openai")
    start = time.time()
    result = await asyncio.to_thread(clip.encode_text, f"test {n}")
    elapsed = time.time() - start
    print(f"Request {n}: {elapsed:.2f}s")
    return result

async def test_concurrent():
    print("Testing 5 concurrent requests...")
    start = time.time()
    results = await asyncio.gather(*[encode_test(i) for i in range(5)])
    total = time.time() - start
    print(f"Total time: {total:.2f}s")
    print(f"Average per request: {total/5:.2f}s")
    print(f"Throughput: {5/total:.2f} req/s")

asyncio.run(test_concurrent())