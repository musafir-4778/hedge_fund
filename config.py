"""config.py — Master Configuration, Quota Guardrails & Smart Cascading Model Tiering.

Provides multi-model tiering (failover ladder across free Google AI Studio models),
remembers daily-exhausted models to immediately route to working tiers,
thread-safe rate limiting (<= 9 RPM), realistic Indian delivery trading taxes,
and SEBI/portfolio risk parameters.
"""

import os
import time
import threading
import logging
from typing import Any, Optional, List, Set
from dotenv import load_dotenv

# Load environment variables explicitly from this directory's .env
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)

# Logger setup
logger = logging.getLogger("wealth_agent_office")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
    )

# Suppress noisy third-party HTTP & generation logs globally
for _noisy_logger in [
    "httpx", "httpcore", "urllib3", "google_genai", "google.genai", 
    "google", "google.ai.generativelanguage", "langchain", "langchain_google_genai"
]:
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

# ---------------------------------------------------------
# API Keys & Multi-Key Pool
# ---------------------------------------------------------
def _load_google_api_keys() -> List[str]:
    keys = []
    import re
    for var in ["GOOGLE_API_KEYS", "GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMINI_API_KEYS"]:
        val = os.getenv(var, "")
        if val:
            for item in re.split(r"[,;\s\n]+", val):
                k = item.strip().strip('"').strip("'")
                if k and k not in keys:
                    keys.append(k)
    for env_var, env_val in os.environ.items():
        if re.match(r"^(GOOGLE_API_KEY_|GEMINI_API_KEY_)\d+$", env_var, re.I):
            k = env_val.strip().strip('"').strip("'")
            if k and k not in keys:
                keys.append(k)
    return keys

GOOGLE_API_KEYS = _load_google_api_keys()
GOOGLE_API_KEY = GOOGLE_API_KEYS[0] if GOOGLE_API_KEYS else ""
_current_key_index = 0
_key_lock = threading.Lock()

def get_active_google_api_key() -> str:
    """Returns the currently active API key from the multi-key pool."""
    global _current_key_index
    with _key_lock:
        if not GOOGLE_API_KEYS:
            return ""
        return GOOGLE_API_KEYS[_current_key_index % len(GOOGLE_API_KEYS)]

def rotate_to_next_google_api_key() -> bool:
    """Rotates to next available API key if daily quota is reached on current key."""
    global _current_key_index
    with _key_lock:
        if len(GOOGLE_API_KEYS) <= 1:
            return False
        _current_key_index = (_current_key_index + 1) % len(GOOGLE_API_KEYS)
        new_key = GOOGLE_API_KEYS[_current_key_index]
        logger.warning(
            f"[API Key Pool] Daily quota exhausted on previous key. Rotated to key #{_current_key_index + 1}/{len(GOOGLE_API_KEYS)} "
            f"({new_key[:6]}...{new_key[-4:]}). Resetting model exhaustion cache."
        )
        with _tier_lock:
            _daily_exhausted_models.clear()
        return True

INITIAL_CAPITAL_INR = float(os.getenv("INITIAL_CAPITAL_INR", 1_000_000.0))

# ---------------------------------------------------------
# Indian Market & Benchmark Constants
# ---------------------------------------------------------
TIMEZONE = "Asia/Kolkata"
CURRENCY = "INR"
BENCHMARK_TICKER = "^NSEI"           # Nifty 50 Index
BROAD_BENCHMARK_TICKER = "^CRSLDX"   # Nifty 500 Index
VIX_TICKER = "^INDIAVIX"             # India VIX volatility gauge

# ---------------------------------------------------------
# Free API Rate-Limit & Docket Controls
# ---------------------------------------------------------
MIN_SECONDS_BETWEEN_LLM_CALLS = float(os.getenv("MIN_SECONDS_BETWEEN_LLM_CALLS", "5.2"))  # ~11.5 RPM (Safe 25% buffer below 15 RPM free tier)
MAX_DEBATE_ROUNDS = 1                # Keeps per-ticker call count ~10 calls
MAX_DAILY_DOCKET_SIZE = 5            # Max 5 new stocks evaluated per day

# ---------------------------------------------------------
# Portfolio Concentration & Risk Guardrails
# ---------------------------------------------------------
MAX_ACTIVE_POSITIONS = 15
MAX_SINGLE_STOCK_PCT = 0.12          # Max 12% in single stock
MAX_SECTOR_EXPOSURE_PCT = 0.30       # Max 30% in single sector
MAX_PORTFOLIO_RISK_PER_TRADE = 0.015 # Max 1.5% total NAV loss if stop-loss triggers
ATR_STOP_MULTIPLIER = 3.0            # Hard stop = Entry - (3.0 * ATR_14) to prevent premature whipsaws on multi-month winners
MIN_RISK_REWARD_RATIO = 2.5          # (Target - Entry) / (Entry - Stop) >= 2.5
DEFAULT_HOLDING_DAYS = 30            # Time-stop review window

# ---------------------------------------------------------
# Liquid Mutual Fund & Risk-Free Treasury Parameters
# ---------------------------------------------------------
LIQUID_FUND_ANNUAL_YIELD = 0.0675     # 6.75% annualized risk-free liquid fund return
LIQUID_SWEEP_CASH_RATIO = 0.30        # Sweep idle cash when cash exceeds 30% NAV
LIQUID_OPERATIONAL_BUFFER_PCT = 0.10  # Maintain 10% NAV operational cash buffer

# ---------------------------------------------------------
# Realistic Indian Delivery Trade Cost Model
# ---------------------------------------------------------
STT_PCT = 0.001                      # 0.10% Securities Transaction Tax on Buy & Sell (Delivery)
EXCHANGE_AND_SEBI_PCT = 0.000035     # ~0.0035% NSE transaction + SEBI turnover charges
STAMP_DUTY_BUY_PCT = 0.00015         # 0.015% Stamp duty on Buy side only
ESTIMATED_SLIPPAGE_PCT = 0.0005      # 0.05% market impact / bid-ask slippage

# ---------------------------------------------------------
# Database Path
# ---------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "paper_portfolio_india.db")
NIFTY500_CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "ind_nifty500list.csv")

# ---------------------------------------------------------
# Multi-Model Tiering Ladders (Free-Tier Cascading Failover)
# ---------------------------------------------------------
def _parse_tier_list(env_var: str, default: List[str]) -> List[str]:
    raw = os.getenv(env_var, "")
    if raw:
        items = [x.strip() for x in raw.split(",") if x.strip()]
        if items:
            return items
    return default

DEFAULT_QUICK_TIERS = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
]
DEFAULT_DEEP_TIERS = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
]

# Honor QUICK_THINK_MODEL and DEEP_THINK_MODEL from .env if provided
_env_quick = os.getenv("QUICK_THINK_MODEL")
if _env_quick:
    if _env_quick in DEFAULT_QUICK_TIERS:
        DEFAULT_QUICK_TIERS.remove(_env_quick)
    DEFAULT_QUICK_TIERS.insert(0, _env_quick)

_env_deep = os.getenv("DEEP_THINK_MODEL")
if _env_deep:
    if _env_deep in DEFAULT_DEEP_TIERS:
        DEFAULT_DEEP_TIERS.remove(_env_deep)
    DEFAULT_DEEP_TIERS.insert(0, _env_deep)

QUICK_THINK_TIERS = _parse_tier_list("QUICK_THINK_TIERS", DEFAULT_QUICK_TIERS)
DEEP_THINK_TIERS = _parse_tier_list("DEEP_THINK_TIERS", DEFAULT_DEEP_TIERS)

# Track models that have exceeded daily quota (429 RESOURCE_EXHAUSTED)
_daily_exhausted_models: Set[str] = set()
_tier_lock = threading.Lock()

# ---------------------------------------------------------
# High-Throughput Sliding-Window Token Bucket Rate Limiter
# ---------------------------------------------------------
import collections

class SlidingWindowRateLimiter:
    """Thread-safe sliding-window rate limiter ensuring <= max_rpm (e.g. 11 RPM) with safe buffer."""
    def __init__(self, max_rpm: int = 11, min_gap_seconds: float = 5.2):
        self.max_rpm = max_rpm
        self.min_gap_seconds = min_gap_seconds
        self.timestamps: collections.deque = collections.deque()
        self.lock = threading.Lock()
        self.last_start_time: float = 0.0

    def acquire(self):
        with self.lock:
            now = time.time()
            # Purge timestamps older than 60 seconds
            while self.timestamps and (now - self.timestamps[0]) > 60.0:
                self.timestamps.popleft()

            # Enforce minimum gap between consecutive calls (absorbs wire time)
            time_since_last = now - self.last_start_time
            if time_since_last < self.min_gap_seconds:
                sleep_gap = self.min_gap_seconds - time_since_last
                time.sleep(sleep_gap)
                now = time.time()

            # Enforce 60-second window capacity (max 11 calls)
            if len(self.timestamps) >= self.max_rpm:
                oldest = self.timestamps[0]
                sleep_window = 60.0 - (now - oldest) + 0.2
                if sleep_window > 0:
                    time.sleep(sleep_window)
                    now = time.time()

            self.timestamps.append(now)
            self.last_start_time = now

_rate_limiter = SlidingWindowRateLimiter(max_rpm=11, min_gap_seconds=MIN_SECONDS_BETWEEN_LLM_CALLS)


def enforce_llm_pacing():
    """Thread-safe pacing ensuring <= 11.5 RPM via sliding window token bucket."""
    _rate_limiter.acquire()


def get_active_model_for_tier(is_deep_think: bool = False) -> str:
    """Returns the first non-exhausted model in the tier ladder."""
    ladder = DEEP_THINK_TIERS if is_deep_think else QUICK_THINK_TIERS
    with _tier_lock:
        for m in ladder:
            if m not in _daily_exhausted_models:
                return m
        # If all exhausted, reset to first
        logger.warning("[Model Tiering] All models in ladder recorded quota exhaustion. Resetting ladder.")
        _daily_exhausted_models.clear()
        return ladder[0]


def get_quick_think_llm(temperature: float = 0.1, model_override: Optional[str] = None):
    """Returns ChatGoogleGenerativeAI for high-RPM nodes using active available tier model."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    model_name = model_override or get_active_model_for_tier(is_deep_think=False)
    api_key = get_active_google_api_key()
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key,
        timeout=25.0,
        max_retries=1,
    )


def get_deep_think_llm(temperature: float = 0.1, model_override: Optional[str] = None):
    """Returns ChatGoogleGenerativeAI for synthesis nodes using active available tier model."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    model_name = model_override or get_active_model_for_tier(is_deep_think=True)
    api_key = get_active_google_api_key()
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key,
        timeout=35.0,
        max_retries=1,
    )


def is_auth_error(err: Exception) -> bool:
    """Detects invalid, deleted, or unauthenticated API keys (HTTP 401)."""
    msg = str(err).lower()
    return any(ind in msg for ind in ["401", "unauthenticated", "account_state_invalid", "api_key_invalid", "deleted or disabled"])


def is_network_error(err: Exception) -> bool:
    """Detects transient network/DNS dropouts (e.g. getaddrinfo failed, connection reset)."""
    msg = str(err).lower()
    indicators = [
        "getaddrinfo", "name resolution", "dns", "connection reset",
        "remotely closed", "network is unreachable", "connection aborted",
        "temporary failure in name resolution", "errno 11001", "winerror 10054"
    ]
    return any(ind in msg for ind in indicators)


def is_per_minute_rate_limit(err: Exception) -> bool:
    """Detects if error is a transient per-minute rate limit (RPM 429) rather than daily limit."""
    msg = str(err).lower()
    return ("per minute" in msg or "rpm" in msg or ("429" in msg and "per day" not in msg and "daily" not in msg))


def is_daily_quota_error(err: Exception) -> bool:
    """Detects if error is true daily quota exhaustion (RPD limit)."""
    msg = str(err).lower()
    return ("per day" in msg or "daily" in msg or "rpd" in msg)


def is_cascadable_error(err: Exception) -> bool:
    """Detects any server timeout, capacity spike, 404 deprecation, or quota error that warrants tier cascading."""
    msg = str(err).lower()
    indicators = [
        "504", "deadline", "timeout", "timed out",
        "503", "unavailable", "high demand", "capacity",
        "502", "bad gateway", "connection",
        "429", "resource_exhausted", "quota",
        "404", "not_found", "not found", "no longer supported"
    ]
    return any(ind in msg for ind in indicators)


def paced_llm_invoke(
    llm: Any,
    prompt_or_messages: Any,
    is_deep_think: bool = False,
    schema: Optional[Any] = None,
    temperature: Optional[float] = None
) -> Any:
    """Thread-safe paced invocation with key pooling and instant failover across model tiers.

    Preserves node-level temperature across tier failovers and exhausts each model
    across all available API keys before cascading to the next model tier.
    """
    ladder = DEEP_THINK_TIERS if is_deep_think else QUICK_THINK_TIERS
    num_keys = len(GOOGLE_API_KEYS) or 1
    target_temp = temperature if temperature is not None else getattr(llm, "temperature", 0.1)

    for attempt in range(5):
        for current_model in ladder:
            if current_model in _daily_exhausted_models:
                continue

            model_quota_exhausted = False
            model_deprecated = False

            for key_idx in range(num_keys):
                enforce_llm_pacing()
                try:
                    active_llm = (
                        get_deep_think_llm(temperature=target_temp, model_override=current_model)
                        if is_deep_think
                        else get_quick_think_llm(temperature=target_temp, model_override=current_model)
                    )
                    if schema is not None:
                        active_llm = active_llm.with_structured_output(schema)

                    return active_llm.invoke(prompt_or_messages)
                except Exception as e:
                    err_msg = str(e).lower()
                    if is_auth_error(e):
                        logger.warning(f"[API Key Pool] Current API key returned auth error ({e}). Rotating to next key...")
                        rotate_to_next_google_api_key()
                        continue
                    elif is_daily_quota_error(e):
                        logger.warning(
                            f"[API Key Pool] Model '{current_model}' daily quota reached on key #{_current_key_index+1}/{num_keys} ({e}). "
                            f"Rotating to next key..."
                        )
                        if num_keys > 1:
                            rotate_to_next_google_api_key()
                            if key_idx == num_keys - 1:
                                model_quota_exhausted = True
                            continue
                        else:
                            model_quota_exhausted = True
                            break
                    elif is_per_minute_rate_limit(e):
                        logger.warning(f"[Model Tiering] Model '{current_model}' hit transient RPM limit. Pausing 6.0s...")
                        time.sleep(6.0)
                        continue
                    elif is_network_error(e):
                        logger.warning(f"[Network Guard] Connection dropout detected ({e}). Pausing 10.0s...")
                        time.sleep(10.0)
                        continue
                    elif "404" in err_msg or "not_found" in err_msg or "no longer available" in err_msg:
                        logger.warning(f"[Model Tiering] Model '{current_model}' deprecated or not found (404). Cascading...")
                        model_deprecated = True
                        break
                    elif is_cascadable_error(e):
                        logger.warning(f"[Model Tiering] Model '{current_model}' encountered transient spike/timeout ({e}). Trying next key...")
                        rotate_to_next_google_api_key()
                        time.sleep(2.0)
                        continue
                    else:
                        logger.warning(f"[Model Tiering] Model '{current_model}' encountered unexpected error ({e}). Trying next key...")
                        rotate_to_next_google_api_key()
                        time.sleep(1.0)
                        continue

            # Only permanently blacklist if true daily quota or 404 deprecation
            if model_quota_exhausted:
                with _tier_lock:
                    _daily_exhausted_models.add(current_model)
                    logger.warning(f"[Model Tiering] Model '{current_model}' daily quota exhausted across all {num_keys} keys.")
            elif model_deprecated:
                with _tier_lock:
                    _daily_exhausted_models.add(current_model)
                    logger.warning(f"[Model Tiering] Model '{current_model}' deprecated. Excluded from future passes.")

        if attempt < 4:
            sleep_time = (attempt + 1) * 8.0
            logger.info(f"[Model Tiering] Pass {attempt+1} encountered transient timeouts across tiers. Pausing {sleep_time:.0f}s before retry...")
            time.sleep(sleep_time)

    # If all tiers and keys exhausted
    raise RuntimeError(f"All free-tier models in ladder {ladder} have exhausted daily quotas across all {num_keys} keys.")


def extract_text_content(response: Any) -> str:
    """Safely extracts text string from LLM response whether str, list of chunks, or AIMessage."""
    if hasattr(response, "content"):
        content = response.content
    else:
        content = response

    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(p.get("text", ""))
            elif hasattr(p, "text"):
                parts.append(str(p.text))
            else:
                parts.append(str(p))
        return "".join(parts)
    return str(content)

