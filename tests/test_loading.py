import redis

# Create a Redis client
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Test connection
r.set('test_key', 'Hello, Redis!')
value = r.get('test_key')
print(value)  # Should print: Hello, Redis!