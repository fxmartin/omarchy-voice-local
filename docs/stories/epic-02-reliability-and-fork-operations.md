# Epic 2: Reliability and Fork Operations

## Epic Overview
**Epic ID**: Epic-02
**Description**: Keeping the fork trustworthy in daily use: offline CI on the self-hosted GitLab, launches that behave like the user's own, a daemon that recovers from network drops without leaving the microphone open, and an installer that sets up every shipped component.
**Business Value**: The mute promise holds, apps opened by voice behave like apps opened by hand, and every change is verified before it lands.
**Success Metrics**: Pipelines are green on `main`. After a network drop the daemon either reconnects or reports an error, and no recorder outlives a reported mute. A fresh install shows the orb without manual steps.

## Epic Scope
**Total Stories**: 5 | **Total Points**: 14 | **MVP Stories**: 3

## Features in This Epic

### Feature 2.1: Delivered

#### Stories

##### Story 2.1-001: Run the GitHub checks on self-hosted GitLab
**Status**: Done
**User Story**: As a maintainer, I want every push to the GitLab remote verified offline so that nothing merges on "works on my machine".
**Priority**: P0
**Story Points**: 3

**Acceptance Criteria**:
- **Given** a push to `main` **When** the pipeline runs **Then** the publication history scan, bash syntax check, unit suite on Python 3.11 and 3.14, and package build with entry point checks all run offline as root on arm64.
- **Given** a cache miss **When** a job installs **Then** it fails instead of reaching the network.

**Technical Notes**: `.gitlab-ci.yml`, `uv.lock`, and `ci/Containerfile`. Delivered in commit 842622f.

**Definition of Done**:
- [x] Code implemented and peer reviewed
- [x] Tests written and passing
- [x] User-facing docs updated in the same commit for behavior-changing diffs

**Dependencies**: none
**Risk Level**: Low

##### Story 2.1-002: Isolate desktop action tests from installed browsers
**Status**: Done
**User Story**: As a maintainer, I want the suite to pass whether or not Google Chrome is installed so that tests do not depend on the machine running them.
**Priority**: P1
**Story Points**: 1

**Acceptance Criteria**:
- **Given** a machine without Google Chrome **When** the desktop action tests run **Then** they pass using a synthetic desktop entry.

**Technical Notes**: `tests/test_policy.py`. Delivered in commit a4d76e9.

**Definition of Done**:
- [x] Code implemented and peer reviewed
- [x] Tests written and passing
- [x] User-facing docs updated in the same commit for behavior-changing diffs

**Dependencies**: none
**Risk Level**: Low

##### Story 2.1-003: Start every app the assistant opens in the desktop session
**Status**: Done
**User Story**: As a user, I want browsers and apps opened by voice to run like apps I open myself so that Chromium stops crashing.
**Priority**: P0
**Story Points**: 3

**Acceptance Criteria**:
- **Given** any launch path, such as desktop apps, URLs, `omarchy launch`, compose panes, terminals, and web windows **When** the assistant opens something **Then** it starts as a user-manager service outside the daemon's sandbox.
- **Given** a failed launch **When** it returns **Then** its error text and exit status still reach the model.

**Technical Notes**: `USER_MANAGER` and `Executor._launch` in `src/omarchy_voice/tools.py`. New launch paths must use `_launch`. Delivered in commits 0fa3b33 and 6de70dd.

**Definition of Done**:
- [x] Code implemented and peer reviewed
- [x] Tests written and passing
- [x] User-facing docs updated in the same commit for behavior-changing diffs

**Dependencies**: none
**Risk Level**: Medium

### Feature 2.2: Open Defects

#### Stories

##### Story 2.2-001: Recover from network drops without leaving the microphone open
**Status**: Done
**User Story**: As a user, I want the Realtime engine to recover or say so after a network drop so that it never sits silent with a recorder still running.
**Priority**: P0
**Story Points**: 5

**Acceptance Criteria**:
- **Given** the websocket closes on a keepalive timeout **When** the daemon reconnects **Then** the new session answers the next utterance, or the state shows error and the user is told.
- **Given** a reconnected session that stops responding **When** a response or health check times out **Then** the daemon retries within its reconnect budget instead of idling with no connection.
- **Given** the log reports "mic stopped" or the gate reports muted **When** checked **Then** no recorder process from any session is still running, asserted by tests across the reconnect path.
- **Given** a drop mid-utterance **When** recovery completes **Then** the bar and orb reflect the true state within a few seconds.

**Technical Notes**: Observed live: after a hotspot stall, a reconnected session transcribed a command, never replied, and then held no connection while reporting idle, and a recorder started before the drop outlived two logged mic stops. Reproduce with the fake websocket server in `tests/test_realtime_wire.py` by dropping keepalives. The reconnect loop and `_mic_loop` are in `src/omarchy_voice/realtime.py`.

**Definition of Done**:
- [ ] Code implemented and peer reviewed
- [ ] Tests written and passing
- [ ] User-facing docs updated in the same commit for behavior-changing diffs

**Dependencies**: none
**Risk Level**: High

##### Story 2.2-002: Install and enable the orb overlay
**Status**: Done
**User Story**: As a user, I want the installer to set up the listening orb so that I see it without copying plugin files by hand.
**Priority**: P2
**Story Points**: 2

**Acceptance Criteria**:
- **Given** the installer's desktop integration step **When** the user accepts **Then** `voice.orb` is copied beside the bar widget and enabled with `omarchy plugin enable`.
- **Given** the shell has not yet seen the new plugin **When** enabling it **Then** the installer rescans or restarts the shell without resetting the user's bar layout.
- **Given** `uninstall.sh` **When** run **Then** it removes and disables the orb as it does the bar widget.

**Technical Notes**: Never call `omarchy-refresh-shell`: it resets `shell.json` to defaults and disables third-party plugins. Use `omarchy-restart-shell` or the shell's plugin rescan.

**Definition of Done**:
- [ ] Code implemented and peer reviewed
- [ ] Tests written and passing
- [ ] User-facing docs updated in the same commit for behavior-changing diffs

**Dependencies**: none
**Risk Level**: Low
