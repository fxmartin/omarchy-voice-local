# Local engine

The local engine keeps your voice on your machine. The microphone is endpointed
locally, transcribed by a whisper.cpp server bound to loopback, and answered by a
local Piper voice. Only the recognised text goes to the planner, which starts on
OpenAI chat and can move to a local model with a config change.

## Requirements

- whisper.cpp with its Vulkan backend, run as a user service. See below.
- Piper and one voice. See below.
- PipeWire's `pw-record` and `pw-cat`, which the other engines also use.
- A planner endpoint. By default this is OpenAI chat with `OPENAI_API_KEY` from
  `~/.config/omarchy-voice/env`, the same key the other engines use.

## Select the engine

Try it in the foreground first. Stop the running daemon, then start the local
engine:

```sh
omarchy-voice listen quit
omarchy-voice run --engine local
```

To choose it persistently, set the engine in `~/.config/omarchy-voice/config.toml`
and restart the user service:

```toml
[openai]
engine = "local"
```

```sh
systemctl --user restart omarchy-voice
```

Set `engine = "realtime"` or `"live"` to switch back. Listening, the bar widget,
the orb, `listen confirm`, `listen cancel` and `listen say` behave the same in
every engine.

## Speech recognition: whisper.cpp

Install whisper.cpp and its Vulkan backend from the Arch repositories. The Vulkan
backend runs recognition on the integrated GPU.

```sh
sudo pacman -S whisper-cpp ggml-vulkan
```

Download a model into the directory the service reads from:

```sh
mkdir -p ~/.local/share/omarchy-voice/models
curl -L -o ~/.local/share/omarchy-voice/models/ggml-small.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin
```

If another tool has already downloaded ggml models, such as voxtype, you can
point the service at those files instead of downloading again.

Install and start the server as a user service. It listens on `127.0.0.1:9000`
only, because that port receives your voice.

```sh
cp ~/.local/share/omarchy-voice/share/whisper-server.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now whisper-server
```

To use a different model, override the path without editing the shipped file:

```sh
systemctl --user edit whisper-server
```

```ini
[Service]
Environment=WHISPER_MODEL=%h/.local/share/omarchy-voice/models/ggml-base.en.bin
```

### Choosing a model

Latency matters more than accuracy for short commands. These times were measured
on a Core Ultra 7 258V with Arc 140V graphics through the Vulkan backend. Each is
the median of five runs of a 2.4 second command, sent through the engine's own
recognition client after a warm-up. Every model transcribed the command
correctly.

| Model | Size | Median latency |
|---|---|---|
| `base.en` | 142 MB | 0.21 s |
| `small.en` | 466 MB | 0.51 s |
| `medium.en` | 1.5 GB | 1.23 s |
| `large-v3-turbo`, q5_0 | 547 MB | 1.69 s |
| `large-v3-turbo` | 1.6 GB | 1.75 s |

The service defaults to `small.en`, the largest model that answers in under a
second. Use `base.en` on slower machines. The `.en` models recognise English
only; use a multilingual model such as `small` and set `[local] language` for
other languages.

`omarchy-voice doctor` checks that the server answers and reports the model the
service is configured to load. It warns when `[local] stt_url` is not a loopback
address, because audio would then leave the machine.

## Speech output: Piper

Install Piper with uv. Do not use `pacman -S piper`: the Arch package of that
name is an unrelated tool for configuring gaming mice.

```sh
uv tool install piper-tts
```

Download a voice and its config file. Both must sit side by side, with the
config named after the model plus `.json`.

```sh
mkdir -p ~/.local/share/omarchy-voice/voices
cd ~/.local/share/omarchy-voice/voices
base=https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium
curl -LO "$base/en_US-lessac-medium.onnx"
curl -LO "$base/en_US-lessac-medium.onnx.json"
```

Point the config at the model. `~` is not expanded, so use the full path:

```toml
[local]
piper_model = "/home/you/.local/share/omarchy-voice/voices/en_US-lessac-medium.onnx"
```

Other voices and languages are listed in the
[Piper voice catalogue](https://huggingface.co/rhasspy/piper-voices). Replies
are spoken sentence by sentence, so the first sentence starts while the rest
are still being synthesized. With `en_US-lessac-medium` each sentence takes
about 1.2 seconds to synthesize on the reference laptop, most of it loading the
voice.

`omarchy-voice doctor` checks the Piper binary, the voice and its config, and
synthesizes a short test phrase without playing it. If Piper or the voice is
missing at runtime, replies are shown as notifications instead.

## Configuration

Every key under `[local]` is optional. The
[configuration example](../share/config.example.toml) lists them with defaults.

| Key | Default | Purpose |
|---|---|---|
| `stt_url` | `http://127.0.0.1:9000` | whisper.cpp server |
| `stt_timeout_seconds` | `20` | Give up on one utterance after this long |
| `language` | `en` | Recognition language |
| `piper_model` | unset | Piper voice `.onnx` file |
| `planner_base_url` | `https://api.openai.com/v1` | Any OpenAI-compatible chat endpoint |
| `planner_model` | `gpt-4.1` | Planner model |
| `planner_api_key_env` | `OPENAI_API_KEY` | Variable holding the key; empty sends none |
| `history_turns` | `10` | Earlier utterances the planner remembers |
| `endpoint_silence_ms` | `700` | Silence that ends an utterance |
| `endpoint_min_speech_ms` | `250` | Shorter speech is ignored as noise |
| `endpoint_max_speech_ms` | `15000` | Longer speech is cut and sent |
| `endpoint_preroll_ms` | `300` | Audio kept from just before speech began |

To move the planner to a local server, point it at any OpenAI-compatible
endpoint and send no key. For Ollama:

```toml
[local]
planner_base_url = "http://127.0.0.1:11434/v1"
planner_model = "qwen3:8b"
planner_api_key_env = ""
```

Small local models are weaker at choosing desktop tools than the default.

## Privacy

- The microphone opens only while listening is toggled on. Toggling off stops
  the recorder, so nothing is captured while muted.
- Audio goes only to the whisper.cpp server at `stt_url`. It is held in memory
  and never written to disk or the log. Doctor warns if `stt_url` is not a
  loopback address.
- The recognised text, the planner's tool calls and their results go to the
  planner endpoint. With the default settings that is OpenAI.
- Conversation memory for follow-ups is kept in memory for the session only and
  is never saved.
- Camera vision and durable task workers keep their own configured providers.
  The local engine does not change where they send data.

## Limitations

- Speech is half duplex. The microphone ignores what it hears while the
  assistant speaks, so you cannot interrupt it. Set `barge_in = true` under
  `[ears]` only with headphones or PipeWire echo cancellation.
- Endpointing is energy based. In a noisy room, raise `endpoint_min_speech_ms`
  or lower `endpoint_silence_ms` if utterances run together or get cut short.
- The `.en` recognition models understand English only.
- Each Piper sentence starts a new process that reloads the voice, which adds
  about a second before the first words.
- A held action is released only by saying one of the `confirm_words`, or by
  `omarchy-voice listen confirm`. The planner cannot confirm on your behalf.
- If the recognition server, the planner or Piper fails, that turn fails with a
  notification naming the stage, and listening continues.
