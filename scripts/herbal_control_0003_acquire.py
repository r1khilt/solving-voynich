"""Finish the frozen Commons source set in finite, rate-spaced batches.

The underlying fetcher enforces the 216-page panel, 5 MiB per JPEG and 150
MiB total cap. This controller makes at most 30 eight-image attempts over at
most eight hours, with a 12-minute gap between successful batches and a
20-minute cooldown after HTTP 429. Repeated 429s stop the run.
"""

from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import time

from herbal_control_0003_fetch import RateLimited, run


LOCK = Path("/tmp/herbal_control_0003_acquire.lock")
SUCCESS_COOLDOWN = 12 * 60
RATE_LIMIT_COOLDOWN = 20 * 60
MAX_ATTEMPTS = 30
MAX_RATE_LIMITS = 6
MAX_SECONDS = 8 * 60 * 60


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-batch-now", action="store_true",
                        help="use only when a prior acquisition batch ended over 12 minutes ago")
    args = parser.parse_args()
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic() + MAX_SECONDS
        attempts = 0
        rate_limits = 0
        cooldown = 0 if args.first_batch_now else SUCCESS_COOLDOWN
        while attempts < MAX_ATTEMPTS and time.monotonic() + cooldown < deadline:
            if cooldown:
                print(f"cooldown {cooldown}s before next bounded source batch", flush=True)
                time.sleep(cooldown)
            attempts += 1
            try:
                manifest = run(limit=8)
            except RateLimited as exc:
                rate_limits += 1
                print(f"attempt {attempts}: {exc}; rate limits {rate_limits}/{MAX_RATE_LIMITS}",
                      flush=True)
                if rate_limits >= MAX_RATE_LIMITS:
                    break
                cooldown = RATE_LIMIT_COOLDOWN
                continue
            pages = manifest["recorded_source_pages"]
            print(f"attempt {attempts}: {pages}/216 source pages, "
                  f"{manifest['recorded_total_bytes']} bytes", flush=True)
            if pages == 216:
                return
            cooldown = SUCCESS_COOLDOWN
        print(f"bounded acquisition stopped: {attempts} attempts, {rate_limits} rate limits",
              flush=True)


if __name__ == "__main__":
    main()
