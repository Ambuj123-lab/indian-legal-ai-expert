from slowapi import Limiter
from slowapi.util import get_remote_address

# Global Rate Limiter
# Uses remote address (IP) as the key
limiter = Limiter(key_func=get_remote_address)
