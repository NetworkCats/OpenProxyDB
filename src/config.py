"""Configuration constants for OpenProxyDB crawler."""

# API Endpoints
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
ZH_WIKIPEDIA_API_URL = "https://zh.wikipedia.org/w/api.php"

# API Parameters
LOCAL_BLOCKS_PARAMS = {
    "action": "query",
    "list": "blocks",
    "bkprop": "user|timestamp|expiry|reason|range",
    "bklimit": 500,
    "format": "json",
}

ZH_LOCAL_BLOCKS_PARAMS = {
    "action": "query",
    "list": "blocks",
    "bkprop": "user|timestamp|expiry|reason|range",
    "bklimit": 500,
    "format": "json",
}

GLOBAL_BLOCKS_PARAMS = {
    "action": "query",
    "list": "globalblocks",
    "bgprop": "target|timestamp|expiry|reason|range",
    "bglimit": 500,
    "format": "json",
}

# Proxy-related keyword to classification mapping
# Keys: keywords to search for in block reasons (case-insensitive)
# Values: classification column names for CSV output
PROXY_CLASSIFICATIONS = {
    "anonblock": "anonblock",
    "proxy": "proxy",
    "vpn": "vpn",
    "cdn": "cdn",
    "public wi-fi": "public-wifi",
    "rangeblock": "rangeblock",
    "school block": "school-block",
    "tor": "tor",
    "webhost": "webhost",
    "代理": "proxy",
    "虚拟专用网": "vpn",
    "虛擬專用網": "vpn",
    "托管": "webhost",
    "託管": "webhost",
    "洋葱": "tor",
    "洋蔥": "tor",
    "学校封禁": "school-block",
    "學校封禁": "school-block",
    "段封禁": "rangeblock",
    "段封鎖": "rangeblock",
    "公共无线": "public-wifi",
    "公共無線": "public-wifi",
}

# Rate limiting settings
REQUESTS_PER_SECOND = 10
REDUCED_REQUESTS_PER_SECOND = 2
REQUEST_TIMEOUT = 20
MAX_CONCURRENT_REQUESTS = 3

# Retry settings
MAX_RETRIES = 5
RETRY_DELAY = 5
BACKOFF_MULTIPLIER = 2

# User-Agent template for Wikipedia API compliance
# The actual User-Agent will be built with the aiohttp version at runtime
USER_AGENT_TEMPLATE = (
    "OpenProxyDB/1.0 "
    "(https://github.com/networkcats/OpenProxyDB) "
    "Python-aiohttp/{version}"
)


def get_user_agent() -> str:
    """Get the User-Agent string with the current aiohttp version.

    Returns:
        User-Agent string for API requests.
    """
    import aiohttp
    return USER_AGENT_TEMPLATE.format(version=aiohttp.__version__)

# GitHub repository
GITHUB_OWNER = "networkcats"
GITHUB_REPO = "OpenProxyDB"

# File paths
LAST_CRAWL_TIME_FILE = "last_crawl_time.txt"
CSV_FILENAME = "proxy_blocks.csv"
METADATA_CSV_FILENAME = "block_metadata.csv"
