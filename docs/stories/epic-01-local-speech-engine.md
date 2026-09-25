# Epic 1: Local Speech Engine

## Epic Overview
**Epic ID**: Epic-01
**Description**: A third voice engine, selected with `engine = "local"`, that keeps speech on the machine. The microphone is endpointed locally, transcribed by a whisper.cpp server, planned by the existing tool loop against a configurable OpenAI-compatible endpoint, and answered through streaming Piper speech. The Realtime and Live engines stay available and unchanged; the engine key is the switch.
**Business Value**: Room audio no longer has to stream to a cloud API while listening is on. A network stall costs only the planning step instead of the whole conversation, and the brain can later move to a local model with a config change instead of code.
**Success Metrics**: With `engine = "local"`, a spoken command such as "go to workspace two" completes end to end on the reference laptop with no audio leaving the machine. Toggling between engines needs only a config change and a daemon restart. The offline CI suite covers the engine through fake speech providers.

## Epic Scope
**Total Stories**: 10 | **Total Points**: 34 | **MVP Stories**: 7

## Features in This Epic

### Feature 1.1: Engine Foundation

#### Stories

##### Story 1.1-001: Select the local engine and its configuration
**Status**: Done
**User Story**: As a user, I want to choose `engine = "local"` in my config so that I can switch between cloud and local speech without touching code.
**Priority**: P0
**Story Points**: 3

**Acceptance Criteria**:
- **Given** `engine = "local"` in `config.toml` **When** the daemon starts **Then** it runs the local engine, and `realtime` and `live` still select their existing engines.
- **Given** a `[local]` config section **When** it is loaded **Then** its keys are prefixed like the other engine sections and cover the speech-recognition server URL, recognition language, Piper voice model path, planner base URL, planner model, planner API key variable, and endpointing thresholds, each with a documented default.
- **Given** an unknown or misspelled `[local]` key **When** `omarchy-voice doctor` runs **Then** it is reported the same way unknown keys are today.
- **Given** `omarchy-voice run --engine local` **When** run **Then** the flag overrides the config file.

**Technical Notes**: Extend `Config` and `PREFIXED_SECTIONS` in `src/omarchy_voice/config.py`, the engine checks in `cmd_run` and `cmd_doctor`, and the `--engine` choices in `src/omarchy_voice/cli.py`. Add the section, commented, to `share/config.example.toml`.

**Definition of Done**:
- [ ] Code implemented and peer reviewed
- [ ] Tests written and passing
- [ ] User-facing docs updated in the same commit for behavior-changing diffs

**Dependencies**: none
**Risk Level**: Low

##### Story 1.1-002: Planner with a configurable endpoint and conversation memory
**Status**: Done
**User Story**: As a user, I want the local engine's brain to remember the conversation and talk to any OpenAI-compatible endpoint so that follow-ups work and a local model can replace the cloud one later.
**Priority**: P0
**Story Points**: 3

**Acceptance Criteria**:
- **Given** a configured base URL and API key variable **When** the planner runs **Then** it posts to that endpoint instead of the hardcoded OpenAI URL, and an unset key is only an error when the endpoint requires one.
- **Given** two consecutive utterances in one listening session **When** the second refers to the first, such as "and close it" **Then** the planner receives the earlier exchange, including tool calls and results.
- **Given** a long session **When** the history grows **Then** it is bounded by a configured turn count, oldest turns dropped first, and never persisted to disk.
- **Given** `omarchy-voice say` **When** run with no local settings **Then** it behaves exactly as before.

**Technical Notes**: `src/omarchy_voice/planner.py` hardcodes `CHAT_URL` and rebuilds the message list per turn. Keep the policy gate and pending-confirmation return path intact. A fake HTTP endpoint on a local socket covers the tests.

**Definition of Done**:
- [ ] Code implemented and peer reviewed
- [ ] Tests written and passing
- [ ] User-facing docs updated in the same commit for behavior-changing diffs

**Dependencies**: 1.1-001
**Risk Level**: Low

### Feature 1.2: Hearing

#### Stories

##### Story 1.2-001: Local microphone capture and endpointing
**Status**: Done
**User Story**: As a user, I want the local engine to notice when I start and stop speaking so that each command is transcribed as one utterance without pressing anything.
**Priority**: P0
**Story Points**: 5

**Acceptance Criteria**:
- **Given** listening is toggled on **When** I speak and then pause for the configured silence time **Then** exactly one utterance is handed to recognition, with a short pre-roll so the first syllable is kept.
- **Given** room tone or a brief noise below the minimum utterance length **When** captured **Then** nothing is sent for recognition.
- **Given** an utterance longer than the configured maximum **When** reached **Then** it is cut and sent, so recognition never waits on an open microphone.
- **Given** listening is toggled off or the daemon stops **When** that happens **Then** the recorder process has exited, which a test asserts on every exit path.
- **Given** capture is running **When** frames arrive **Then** the microphone level is published for the orb exactly as the Realtime engine does.

**Technical Notes**: Reuse `pw-record` capture, `frame_level`, and the mute-gate design from `src/omarchy_voice/realtime.py`. Energy-based endpointing first; a neural voice detector is out of scope unless the energy detector measurably misfires. Test with synthetic PCM, no microphone.

**Definition of Done**:
- [ ] Code implemented and peer reviewed
- [ ] Tests written and passing
- [ ] User-facing docs updated in the same commit for behavior-changing diffs

**Dependencies**: 1.1-001
**Risk Level**: Medium

##### Story 1.2-002: whisper.cpp speech recognition client
**Status**: Done
**User Story**: As a user, I want my utterances transcribed by a whisper.cpp server on my own machine so that my voice never leaves it.
**Priority**: P0
**Story Points**: 3

**Acceptance Criteria**:
- **Given** an endpointed utterance **When** it is sent to the configured server **Then** the transcript comes back and is logged as "heard", like the cloud engines.
- **Given** the server is down, slow past its timeout, or returns an error **When** a request fails **Then** the user hears or sees a short failure message and the engine keeps listening.
- **Given** an utterance **When** it is transcribed **Then** its audio is held in memory only and never written to disk.
- **Given** the configured server URL **When** it is not a loopback address **Then** doctor warns that audio leaves the machine.

**Technical Notes**: Use the standard library HTTP client to post a WAV body to the server's inference endpoint, so no new Python dependency is added. Tests use a fake server on a local socket.

**Definition of Done**:
- [ ] Code implemented and peer reviewed
- [ ] Tests written and passing
- [ ] User-facing docs updated in the same commit for behavior-changing diffs

**Dependencies**: 1.1-001
**Risk Level**: Low

##### Story 1.2-003: whisper.cpp server setup with GPU acceleration
**Status**: Done
**User Story**: As a user, I want a documented, checked setup for the whisper.cpp server so that recognition is fast enough to feel like a conversation.
**Priority**: P1
**Story Points**: 3

**Acceptance Criteria**:
- **Given** the setup guide **When** followed on Omarchy **Then** it installs whisper.cpp with the Vulkan backend, fetches a recommended model, and runs the server as a systemd user service bound to loopback only.
- **Given** the local engine is selected **When** doctor runs **Then** it checks the server answers, reports the model it serves, and says how to fix each failure.
- **Given** a short command on the reference laptop **When** timed **Then** recognition latency is recorded in the guide, and the recommended model is the largest one that stays under one second.

**Technical Notes**: Package availability and the exact Vulkan build are the main risk; confirm before committing to a model size. Keep benchmark evidence under ignored `benchmarks/`, never in the repo.

**Definition of Done**:
- [ ] Code implemented and peer reviewed
- [ ] Tests written and passing
- [ ] User-facing docs updated in the same commit for behavior-changing diffs

**Dependencies**: 1.2-002
**Risk Level**: Medium

### Feature 1.3: Speaking

#### Stories

##### Story 1.3-001: Streaming Piper speech through PipeWire
**Status**: Done
**User Story**: As a user, I want replies spoken by a local Piper voice as soon as the first sentence is ready so that the assistant answers without cloud speech.
**Priority**: P0
**Story Points**: 5

**Acceptance Criteria**:
- **Given** a reply **When** it is spoken **Then** synthesis runs sentence by sentence and the first sentence starts playing before the rest is synthesized.
- **Given** audio is playing **When** the engine asks **Then** "is speaking" and the output level are available, so the microphone gate holds while it speaks and the orb breathes with the voice.
- **Given** listening is toggled off, a barge-in, or a new reply **When** speech is playing **Then** playback stops within a quarter second and queued sentences are dropped.
- **Given** Piper or its voice model is missing **When** a reply is due **Then** the reply is shown as a notification and the failure is logged once.

**Technical Notes**: Play through `pw-cat` like `src/omarchy_voice/playback.py`, not the `aplay` path in `src/omarchy_voice/feedback.py`. Tests use a fake synthesizer and a fake player.

**Definition of Done**:
- [ ] Code implemented and peer reviewed
- [ ] Tests written and passing
- [ ] User-facing docs updated in the same commit for behavior-changing diffs

**Dependencies**: 1.1-001
**Risk Level**: Medium

##### Story 1.3-002: Piper voice setup and checks
**Status**: Done
**User Story**: As a user, I want the setup to tell me which Piper voice to install and where so that speech works on the first try.
**Priority**: P1
**Story Points**: 2

**Acceptance Criteria**:
- **Given** the setup guide **When** followed **Then** it names a recommended voice, where to place the model and its config, and how to point the config at them.
- **Given** the local engine is selected **When** doctor runs **Then** it checks the Piper binary, the voice model and its config file, and synthesizes one short phrase without playing it.

**Technical Notes**: The installer may offer the download as an opt-in step, following its existing prompt style; it must never download without asking.

**Definition of Done**:
- [ ] Code implemented and peer reviewed
- [ ] Tests written and passing
- [ ] User-facing docs updated in the same commit for behavior-changing diffs

**Dependencies**: 1.3-001
**Risk Level**: Low

### Feature 1.4: The Engine Loop

#### Stories

##### Story 1.4-001: Local engine session loop
**Status**: Done
**User Story**: As a user, I want to toggle listening, speak a command, and have the local engine act and answer so that it works like the cloud engines from the keyboard, bar, and orb.
**Priority**: P0
**Story Points**: 5

**Acceptance Criteria**:
- **Given** listening is on **When** an utterance is transcribed **Then** it is planned with the full tool set behind the existing policy gate, the actions run, and the reply is spoken.
- **Given** an action that needs confirmation **When** it is held **Then** the confirm and cancel words and the `listen confirm` and `listen cancel` commands behave as they do in the Realtime engine.
- **Given** any stage **When** the state changes **Then** the bar and orb show listening, thinking, acting, confirm, and error exactly as the other engines do.
- **Given** the control socket **When** `listen toggle`, `start`, `stop`, `say`, and `quit` arrive **Then** each works the same as in the Realtime engine.
- **Given** a finished background task **When** its notice arrives **Then** it is announced like the Realtime engine announces it.

**Technical Notes**: A new module beside `realtime.py` and `live.py`. Speech is half duplex by default: the microphone gate holds while speaking unless `barge_in` is on.

**Definition of Done**:
- [ ] Code implemented and peer reviewed
- [ ] Tests written and passing
- [ ] User-facing docs updated in the same commit for behavior-changing diffs

**Dependencies**: 1.1-002, 1.2-001, 1.2-002, 1.3-001
**Risk Level**: Medium

##### Story 1.4-002: Failure handling and privacy guarantees
**Status**: Done
**User Story**: As a user, I want the local engine to degrade clearly when one part fails so that I am never left talking to a silent assistant or an open microphone.
**Priority**: P0
**Story Points**: 3

**Acceptance Criteria**:
- **Given** recognition, planning, or speech fails **When** it happens **Then** the user is told which part failed, the state shows error briefly, and listening resumes without a daemon restart.
- **Given** the network drops **When** a command is planned against a remote endpoint **Then** only that turn fails; recognition and speech keep working.
- **Given** any failure, toggle, or shutdown path **When** it completes **Then** no recorder process is left running, asserted by tests.
- **Given** the log **When** transcripts and errors are written **Then** they go through the existing redaction, and no audio is logged.

**Technical Notes**: Mirror the Realtime engine's restart limits so a missing server cannot crash-loop the systemd unit.

**Definition of Done**:
- [ ] Code implemented and peer reviewed
- [ ] Tests written and passing
- [ ] User-facing docs updated in the same commit for behavior-changing diffs

**Dependencies**: 1.4-001
**Risk Level**: Medium

##### Story 1.4-003: Local engine guide and engine switching docs
**Status**: Done
**User Story**: As a user, I want one guide for setting up and switching to the local engine so that I can move between cloud and local speech with confidence.
**Priority**: P1
**Story Points**: 2

**Acceptance Criteria**:
- **Given** `docs/local.md` **When** read **Then** it covers requirements, whisper.cpp and Piper setup, config keys, switching engines, privacy behavior, and known limitations.
- **Given** the README **When** read **Then** its engine section lists all three engines and links the guide, and stays focused on installation and use.

**Technical Notes**: Follow the structure of `docs/live.md`.

**Definition of Done**:
- [ ] Code implemented and peer reviewed
- [ ] Tests written and passing
- [ ] User-facing docs updated in the same commit for behavior-changing diffs

**Dependencies**: 1.4-001, 1.2-003, 1.3-002
**Risk Level**: Low
