# Sampler persistence and retained server generations

This optional extension is independent of PrefixLM. It applies to ordinary text
causal decoding and integrates with PrefixLM when that attention patch is present.
It does not change the library KV format or turn model snapshots into self-contained
checkpoints. Use the same runtime, model, vocabulary, KV configuration, fixed adapters
and effective sampling configuration when restoring. Cross-version/hardware bitwise
reproducibility is not promised.

## Samplers

`llama_sampler_state_get_size/get_data/set_data` preserve built-in sampler RNG,
adaptation, history and grammar state. Grammar stores rule/element offsets rather
than pointers, including partial UTF-8 and lazy-trigger buffers. Configuration must
match; invalid data leaves the destination unchanged. Backend restore uses existing
graph-safe copy hooks, preserving tensor bindings and committed/transactional state.
Synchronize the attached context before snapshot or restore. Graph output buffers
are not snapshots.

Common sampler snapshots also preserve accepted-token history and the complete
reasoning-budget state machine, including delimiter matchers, UTF-8 waiting and
manual forcing. Grammar and reasoning constraints retain the ordinary host fallback.
Unknown custom samplers reject these APIs: their owners serialize application state
and use existing clone/copy hooks. Encoder/decoder modes have no automatic serializer
for arbitrary user-defined state either.

The built-in sampler format is SMP2; the common bundle version is 2. Sampler and KV
snapshots must describe the same boundary. If sampling already selected the next
token, preserve that pending token instead of sampling it again.

## Server protocol

A single text generation on `/completion`, `/v1/completions`, `/v1/chat/completions`
or `/v1/responses` may set `retain_state: true` and an explicit `id_slot`. When it
reaches `n_predict`, the server preserves KV, sampler state, the selected pending
token and output bookkeeping. Continue with `resume: true`, the original prompt,
matching sampling/parser settings and a new additional `n_predict` budget. Set
`retain_state` again to keep the next boundary. A new user turn is a fresh request.

With `--slot-save-path`, `POST /slots/{id}?action=save` and `restore`, with a
`filename`, persist or restore the retained generation across process restart and
slot-ID remapping. Retained saves use a temporary file plus rename. Slot metadata
uses a -2 tag, byte length and version-3 JSON envelope. Ordinary prompt/KV-only
files retain their existing format and can still be restored; they do not become
resumable generations. Older development generation envelopes are rejected.

Retained slots are excluded from automatic idle-cache eviction and automatic slot
selection. Explicitly replacing, erasing or consuming the retained generation
clears its bookkeeping. Ordinary causal decoding keeps its normal idle prompt/KV
behavior after generation completes. Live adapter mutation is blocked while retained
state exists; restore remains the caller's responsibility to configure identically.

OpenAI parsing is reconstructed from saved emitted text and the original parser
configuration, with stable generated tool-call IDs. Native text and OpenAI streaming
deltas contain only the continuation. Nonstream OpenAI messages and final Responses
objects are cumulative. Each resumed HTTP request has fresh response IDs/framing;
a Responses tool with continuing arguments is reannounced with its logical call ID.
This is generation continuation, not replay of an interrupted network connection.

Retention requires a single text decoder task, explicit slot, fixed adapters and no
speculation. Encoder/encoder-decoder models, noncausal attention, multimodal input,
activated/per-request LoRA, Anthropic transport and parent/child groups reject it.
Ordinary live `n > 1` scheduling remains unchanged. Persisting an entire scheduling
group would be a new generic feature, not encoder/decoder parity.

## Bounded tests

Build the existing tests with `LLAMA_BUILD_TESTS=ON`:

```sh
build/bin/test-chat --parser-continuation
build/bin/test-reasoning-budget
build/bin/test-backend-sampler --model tiny-causal.gguf --device cpu --test host_persistence
build/bin/test-backend-sampler --model tiny-causal.gguf --device cpu --test multi_output_dist_transaction
build/bin/test-backend-sampler --model tiny-causal.gguf --device gpu --test multi_output_dist_transaction
```

The consuming project's HTTP harness additionally checks uninterrupted versus split
generation, pending-token accounting, fresh-process restore, slot remapping, legacy
prompt files and OpenAI parsing on ordinary causal and PrefixLM paths. Qualify CPU
and actual GPU execution on each deployment platform before claiming support there.
