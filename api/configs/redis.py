from upstash_redis import Redis
from api.configs.settings import settings

redis = Redis(
    url=settings.UPSTASH_REDIS_URL, 
    token=settings.UPSTASH_REDIS_TOKEN
)
