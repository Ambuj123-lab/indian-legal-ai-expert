from pybreaker import CircuitBreaker
import asyncio

cb = CircuitBreaker()
print(f"Has call_async: {hasattr(cb, 'call_async')}")
print(f"Dir: {dir(cb)}")
