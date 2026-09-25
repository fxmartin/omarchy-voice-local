"""One-shot OpenAI planner for `omarchy-voice say`.

The daemon itself is speech-to-speech over the Realtime API. This module is
the typed equivalent: the same tools, the same policy gate, no microphone.
It talks to Chat Completions over HTTPS so a command can be tried without
opening a websocket.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from . import capabilities
from .config import Config
from .persona import PERSONA
from .tools import TOOL_SCHEMAS, Executor, tools_for

OPENAI_BASE_URL = "https://api.openai.com/v1"


@dataclass
class Turn:
    """One request and everything that came of it."""
    text: str
    reply: str = ""
    actions: list[str] = field(default_factory=list)
    error: str = ""
    elapsed: float = 0.0
    tokens: dict = field(default_factory=dict)


def to_chat_tools(schemas: list[dict] | None = None) -> list[dict]:
    converted = []
    for schema in schemas if schemas is not None else TOOL_SCHEMAS:
        converted.append({
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["input_schema"],
            },
        })
    return converted


def _system_prompt(config=None) -> str:
    from .tasks import ROUTING
    from .vision import ROUTING as VISION_ROUTING
    return "\n\n".join([
        PERSONA,
        ROUTING if config is not None and config.tasks_enabled else "",
        VISION_ROUTING if config is not None and config.vision_enabled else "",
        capabilities.manifest(),
        "# The desktop right now\n\n" + capabilities.live_state(),
    ])


class PlannerUnavailable(RuntimeError):
    """Something the one-shot planner needs is missing."""


class Planner:
    """Runs spoken instructions through a chat model and the tool executor.

    With no keyword arguments this is the one-shot OpenAI planner behind
    `omarchy-voice say`: no memory, a key is mandatory. `for_local` builds the
    variant for the local engine: any OpenAI-compatible endpoint, an optional
    key, and an in-memory history of the last `history_turns` exchanges.
    """

    def __init__(
        self,
        config: Config,
        executor: Executor,
        *,
        base_url: str = OPENAI_BASE_URL,
        model: str | None = None,
        api_key_env: str | None = None,
        key_required: bool = True,
        history_turns: int = 0,
    ):
        self.config = config
        self.executor = executor
        self.base_url = base_url
        self.model = model or config.planner_model
        self.api_key_env = api_key_env if api_key_env is not None else config.api_key_env
        self.key_required = key_required
        self.history_turns = history_turns
        # One entry per completed utterance, each the messages it added
        # (user, tool calls, tool results, reply). Memory only, never saved.
        self._history: list[list[dict]] = []

    @classmethod
    def for_local(cls, config: Config, executor: Executor) -> "Planner":
        return cls(
            config,
            executor,
            base_url=config.local_planner_base_url,
            model=config.local_planner_model,
            api_key_env=config.local_planner_api_key_env,
            key_required=bool(config.local_planner_api_key_env),
            history_turns=config.local_history_turns,
        )

    def think(self, text: str) -> Turn:
        turn = Turn(text=text)
        started = time.monotonic()
        try:
            turn.reply = self._loop(text, turn)
        except PlannerUnavailable as exc:
            turn.error = str(exc)
            turn.reply = "My planner isn't configured yet."
        except Exception as exc:  # a voice tool must not die on one bad turn
            turn.error = f"{type(exc).__name__}: {exc}"
            turn.reply = "Something went wrong with that."
        turn.elapsed = time.monotonic() - started
        return turn

    def _loop(self, text: str, turn: Turn) -> str:
        key = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
        if not key and self.key_required:
            raise PlannerUnavailable(
                f"{self.api_key_env} is not set — "
                "put it in ~/.config/omarchy-voice/env")

        history = [m for exchange in self._history for m in exchange]
        messages: list[dict] = [
            {"role": "system", "content": _system_prompt(self.config)},
            *history,
            {"role": "user", "content": text},
        ]
        start = len(history) + 1
        tools = to_chat_tools(tools_for(self.config))
        reply = ""

        for _ in range(self.config.max_turns):
            data = _chat(messages, tools, self.config, key,
                         base_url=self.base_url, model=self.model)
            usage = data.get("usage") or {}
            if usage:
                turn.tokens = {
                    "in": usage.get("prompt_tokens", 0),
                    "out": usage.get("completion_tokens", 0),
                }
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            said = (message.get("content") or "").strip()
            if said:
                reply = said
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                reply = reply or "Done."
                self._remember(messages, start, reply)
                return reply

            messages.append({
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": tool_calls,
            })
            for call in tool_calls:
                fn = call.get("function") or {}
                name = fn.get("name", "")
                raw_args = fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError as exc:
                    outcome_text = f"ERROR: could not parse arguments: {exc}"
                else:
                    outcome = self.executor.call(name, args)
                    turn.actions.append(self.executor.describe(name, args))
                    outcome_text = outcome.as_tool_result()
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": outcome_text,
                })
            if self.executor.pending:
                reply = reply or "That needs confirmation."
                self._remember(messages, start, reply)
                return reply

        reply = reply or "Ran out of steps on that one."
        self._remember(messages, start, reply)
        return reply

    def _remember(self, messages: list[dict], start: int, reply: str) -> None:
        """Keep this utterance for the next one, dropping the oldest past the cap."""
        if self.history_turns <= 0:
            return
        exchange = messages[start:]
        if exchange[-1].get("role") != "assistant" or exchange[-1].get("tool_calls"):
            exchange.append({"role": "assistant", "content": reply})
        self._history.append(exchange)
        del self._history[:-self.history_turns]


def _chat(messages: list[dict], tools: list[dict], config: Config, key: str,
          *, base_url: str = OPENAI_BASE_URL, model: str | None = None) -> dict:
    body = json.dumps({
        "model": model or config.planner_model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
    }).encode()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:400]
        raise PlannerUnavailable(f"planner HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise PlannerUnavailable(f"could not reach the planner: {exc.reason}") from exc
