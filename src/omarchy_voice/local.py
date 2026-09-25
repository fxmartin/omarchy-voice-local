"""Local speech engine (`engine = "local"`).

Selection and configuration only for now: the recognition, planning and speech
pipeline is not wired in yet, so `run` refuses to start rather than pretending.
"""

from __future__ import annotations

import sys

from .config import Config


def config_problems(config: Config) -> list[str]:
    problems: list[str] = []
    if not config.local_stt_url:
        problems.append("[local] stt_url is empty")
    if not config.local_planner_base_url:
        problems.append("[local] planner_base_url is empty")
    return problems


def run(config: Config) -> int:
    print("the local engine's audio pipeline is not implemented yet", file=sys.stderr)
    return 1
