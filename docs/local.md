# Local engine

The local engine keeps your voice on your machine. The microphone is endpointed
locally, transcribed by a whisper.cpp server bound to loopback, and answered by a
local Piper voice. Only the recognised text goes to the planner, which starts on
OpenAI chat and can move to a local model with a config change.

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
