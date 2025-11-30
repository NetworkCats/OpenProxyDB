"""Configuration constants for OpenProxyDB crawler."""

# API Endpoints
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

# API Parameters
LOCAL_BLOCKS_PARAMS = {
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
}

# Rate limiting settings
REQUESTS_PER_SECOND = 20
REDUCED_REQUESTS_PER_SECOND = 5
REQUEST_TIMEOUT = 20
MAX_CONCURRENT_REQUESTS = 5

# Retry settings
MAX_RETRIES = 3
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
