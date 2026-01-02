import sys
import time
import random
import subprocess
from pathlib import Path

SCRIPTS = [
    "load_points.py",
    "load_assists.py",
    "load_rebounds.py",
    "load_turnovers.py",
    "load_stocks.py",
]

# tune these if you want
BREAK_BETWEEN_SUCCESS = 10          # seconds between successful scripts
BREAK_AFTER_FAILURE = 20            # seconds before retry after a failure
MAX_RETRIES_PER_SCRIPT = 3          # retries per script
JITTER = 4                          # adds randomness to sleeps to look less bot-like


def sleep_with_jitter(seconds: int):
    time.sleep(seconds + random.uniform(0, JITTER))


def run_fresh_process(script: str, args: list[str]) -> int:
    """
    Runs `python script args...` as a brand new Python process.
    Returns the exit code.
    """
    cmd = [sys.executable, script, *args]
    print(f"\n▶ running fresh: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parent,  # ensures we run from Optimizer root
        text=True,
    )
    return result.returncode


def main():
    if len(sys.argv) < 2:
        print("Usage: python load_all.py TEAM [PLAYER_OFF_NAME ...]")
        sys.exit(1)

    team = sys.argv[1].upper()
    players_off = sys.argv[2:]
    args = [team, *players_off]

    for script in SCRIPTS:
        attempt = 0
        while True:
            attempt += 1

            exit_code = run_fresh_process(script, args)

            if exit_code == 0:
                print(f"✅ {script} succeeded")
                sleep_with_jitter(BREAK_BETWEEN_SUCCESS)
                break

            print(f"❌ {script} failed (exit code {exit_code}) — attempt {attempt}/{MAX_RETRIES_PER_SCRIPT}")

            if attempt >= MAX_RETRIES_PER_SCRIPT:
                print(f"🛑 giving up on {script} after {MAX_RETRIES_PER_SCRIPT} attempts")
                sys.exit(exit_code)

            sleep_with_jitter(BREAK_AFTER_FAILURE)

    print("\n✅ all scripts finished")


if __name__ == "__main__":
    main()

