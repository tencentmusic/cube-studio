import os,redis

# 初始化 Redis 客户端
cache = redis.Redis.from_url(os.getenv('KFJ_CACHE_URL',''))
key = os.getenv('KFJ_RUN_ID','xx')
# 读取之前的值
value = cache.get(key)
value = value.decode('utf-8') if value else 0
print(value)
# 计算新的值
value = int(value)+1
# 设置新的值
cache.set(key, value, ex=60*60*10)
print(value)
