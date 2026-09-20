# PrefixLM attention

`LLAMA_ATTENTION_TYPE_PREFIX_LM` is a decoder attention mode. Unspecified
context attention selects it when the model declares PrefixLM capability.
Explicit causal and non-causal overrides remain available. The implementation
uses existing CPU/GPU attention kernels and masks; it adds request semantics,
not a new HRM graph or Metal kernel.

For a complete prefix ending at position P, a query can attend to valid keys
in its sequence when `key <= query` or both positions are below P. This
implementation evaluates the complete prefix bidirectionally in one physical
batch, then uses causal attention for answer tokens, including answer chunks.
Each recurrent invocation therefore sees the full prefix. Splitting a prefix
into successive token batches is not supported.

## Library contract

```cpp
auto params = llama_context_default_params();
params.attention_type = LLAMA_ATTENTION_TYPE_PREFIX_LM;
params.n_seq_max = 1;
params.n_ctx = 4096;
params.n_batch = params.n_ubatch = 4096;
auto * ctx = llama_init_from_model(model, params);
// Check initialization and every decode result.
int result = llama_decode_prefix(ctx, complete_prompt_batch);
// Sample from the final prefix position, then continue causally:
result = llama_decode(ctx, answer_batch);
```

`llama_decode_prefix` is an explicit new-request operation, even for a
one-token prefix. It validates the entire input before replacing old memory.
Every new chat request supplies all retained conversation tokens, including
previous assistant turns and the generation header. No application-side
attention toggling is needed. `llama_get_attention_type` reports the active
mode. `llama_set_causal_attn` cannot override a PrefixLM context.

The supported contract accepts multiple live token-input sequences, positions starting at zero, and contiguous answer positions.
Unified KV permits multiple sequence IDs per input row when their prefix boundaries,
cached dependencies and preceding batch rows agree. Every shared prefix row must also
have the same complete future prefix rows for each owner. Duplicate owners, divergent
histories and different prefix phases reject before mutation. Shared rows use one
physical cell and publish boundary logits to every owner. Separate-stream coupled
input remains unsupported by the existing sequential batch splitter; use independent
rows or evaluate once and copy instead.
Absent position arrays infer positions independently per sequence. A prefix
call replaces only the sequences present in that call. It may contain several
complete independent prefixes, provided their combined size fits one physical
batch. Answer calls may interleave several sequences. `llama_decode_prefix_mixed` also combines phases: list each newly initialized
sequence ID and its exclusive prefix end, and supply all its prefix tokens in the
same batch. Tokens at or beyond that boundary are causal; unlisted live sequences
continue their answers. The existing mask builder applies the boundary per sequence,
without new attention kernels. The complete prefix must fit both logical
and physical batch capacity. Answer chunks may be split internally because
they are causal. Requested context capacity is enforced even if allocation is
rounded up; `llama_n_ctx_seq` returns the usable PrefixLM capacity. Separate KV
streams divide requested context capacity by `n_seq_max`. Unified KV uses a
shared capacity and admits at most `n_seq_max` live sequences; their physical
KV cells cannot exceed the requested capacity. Full forks share cells in unified KV;
shared cells count once until branches append distinct tokens. Unified KV supports
sparse sequence IDs using the existing library ID range.

Invalid input returns -1 without changing the current request. Backend
failure, exception during execution, or abort clears every sequence in the call;
other sequences retain their KV and phase. Cleared sequences require new complete prefixes. PrefixLM calls synchronize before returning,
restore the internal causal execution mode, and check the abort callback after
synchronization, including on backends that cannot abort mid-compute. Serialize
all context and memory operations; the abort callback's application state may
be signaled safely from another thread.

Public memory clear and full sequence removal invalidate the prefix state.
Removing a causal answer suffix updates the next answer position and preserves
prefix KV. Empty intervals are no-ops. Deleting any part of a retained prefix
or creating a hole in the answer returns false without mutation. Full removal
also accepts a finite upper bound covering the request. Keeping a sequence follows the existing cache stream semantics, with phase
metadata removed only where KV was removed. Full-sequence copy replaces the destination
and copies the prefix boundary and next position. Self-copy is a no-op. Cross-stream
forks use the existing deferred buffer-copy path; state saving flushes pending copies.
Bounded copies [0, end) replace the destination with the complete prefix and any answer
head, on either KV layout. Copying an answer range [begin, end) additionally requires
identical physical KV before begin (currently provable in unified KV). It replaces the
destination through end and discards any later suffix. A truncated source boundary has
no saved logits and clears the destination row. Cross-stream copies reuse the existing
full-buffer copy followed by metadata trimming. Empty ranges are no-ops. Incomplete
prefix copies, shifting and dividing remain rejected. Independently recomputed/restored
sequences are not assumed equivalent merely because their lengths or token IDs match;
use shared input or a fork to establish common dependencies.
Wildcard removal validates all sequences before mutation. Memory edits do not
recompute logits: preserve the logits at the rollback boundary, or decode an answer
token again before sampling. Suffix trimming invalidates the saved boundary-logit row. Capacity admission accounts
for surviving owners when replacing a shared sequence, so replacing one branch cannot
reclaim cells still owned by another.

### Persistence and fixed adapters

Context and per-sequence state APIs, including files, serialize KV plus prefix boundaries
and next positions, plus the last requested boundary-logit row per sequence. Sequence restore remaps the saved ID to its destination. PrefixLM
uses distinct context/sequence format identifiers and file versions; ordinary formats
are unchanged. Load the same model, compatible KV layout/types, LoRA weights/scales and control vectors
before restoring. These are runtime snapshots, not self-contained model checkpoints.
`llama_get_logits_seq` exposes the saved boundary row after restore/fork; it returns
null when the final input row did not request logits or a memory edit invalidated it.
Ordinary batch-output indexing is not reconstructed. Do not replay the last prefix token through causal decode to recover logits.
Model/KV snapshots do not serialize sampler or application-owned objects. As in
ordinary encoder/decoder modes, callers own custom state and may use existing
sampler clone/copy hooks. Built-in sampler snapshots and retained server generations
are supplied by the separate, optional generation-persistence patch.

The envelope validates version, sequence IDs/counts, boundaries, requested capacity and
a stable adapter-weight/scale fingerprint before KV mutation. The fingerprint detects
accidental configuration mismatch; it is not cryptographic authentication. The existing
state API still requires the caller to select the same model weights. KV positions must
be contiguous, unique, start at zero and agree with the saved next position. Header
rejection preserves current state; a failure after KV restoration starts clears the
restore destination (all sequences for full-context restore), preserving unrelated
sequences for sequence restore. A backend failure while flushing pending cross-stream
copies invalidates the whole context, since multiple sequences can be affected. Complete host snapshots (`flags = 0`) are supported.
Partial-memory flags concern unsupported SWA/recurrent memories; on-device snapshots
remain unqualified because the existing device reader commits copies in its destructor,
which needs a separate failure/rollback audit before sharing these guarantees.

Ordinary fixed LoRA adapters and control vectors use the existing graph helpers. Install them before prefill,
or after clearing all live sequences. Reapplying the identical configuration while KV
is live is allowed; changing/removing it is rejected. Saved states compare weights and
scales and effective control-vector values rather than object addresses. Activated
LoRA remains unsupported: it needs explicit semantics for bidirectional prefixes.

Encoder, recurrent/hybrid, diffusion, SWA, multi-position and embedding-output paths
remain outside this contract. Unsupported configurations or entry points fail explicitly.

## Common processing, server and CLI

The common prompt helper submits a complete prefix and rejects incremental
prompt reuse or state saving in PrefixLM mode. Warmup also uses the prefix API.

The server (also used by llama-cli) supports multiple slots with `--parallel`.
Complete prefixes and answer tokens can share a scheduler batch, with room reserved
for a pending complete prefix so ongoing answers do not starve it. Cancellation and decode failures clear only affected slots. Model metadata
selects the attention mode automatically; `--attention prefix-lm` selects it
explicitly and `--attention causal` retains the causal control path. Each new
request reprocesses its entire prompt by default. With `cache_prompt: true`, a slot
keeps one complete-prefix host snapshot, including boundary logits. Only an identical
full token sequence hits; changed or extended prefixes are evaluated in full. A hit
restores prefix KV and samples from the saved boundary row with a fresh sampler and
prompt history. `action=erase` also drops this cache. Parent/child requests currently
use normal prefill/fork scheduling rather than this cache fast path. Cache-enabled
requests use host sampling on both misses and hits (including when backend sampling
was requested), so the first draw follows the same path. This cache is distinct from
retained generation state. Partial/chunk reuse, ordinary RAM prompt caches,
checkpoints and context shifting remain disabled.
Explicit incompatible request options fail. Shared-prefix children (`n > 1`) use full
sequence forks and require sufficient slots. Fixed startup LoRA is supported; per-request
adapter changes and the adapter-mutation endpoint remain rejected. Requests may queue
or run as separate active sequences. Speculation and multimodal input remain unsupported.

Ordinary encoder/decoder slot files do not serialize in-flight parent/child
scheduling groups or arbitrary external application objects. PrefixLM supports the
same live child-fork functionality; a persistent scheduler-group format is outside
this patch. Server PrefixLM slot files require the optional generation-persistence
patch; library context/sequence snapshots are supported independently.

The scheduler keeps every prefix complete within physical batch capacity.
Admission rejects oversized input before clearing the previous slot.
Execution failures terminate the request without retrying smaller prefix
chunks. Cancellation clears the slot before a subsequent request. Context
exhaustion is visible to the caller; prompts are not silently truncated.

## Tests

Existing architecture tests include a bounded PrefixLM suite:

```sh
cmake --build build --target test-llama-archs
build/bin/test-llama-archs --prefix-lm --arch hrm_text --seed 42
build/bin/test-llama-archs --prefix-lm --arch llama --seed 42
```

It exercises CPU and available GPU devices, fused/unfused attention, metadata
defaults and explicit overrides, prefix visibility, answer causality,
chunk/single-token parity, answer rollback, admission, forbidden cache operations, one-token
prefixes, full context, abort/recovery and causal/bidirectional controls.
Independent tiny-model HF logits can be generated within this patch as described below.
Tokenizer and stock-server references remain in the consuming project's harness. Quantized weights, other backends,
additional backend combinations require separate qualification. The bounded suite also
checks full-sequence forks, branch isolation, sparse-ID persistence, fresh-context and
file restore, ID remapping, empty/corrupt/truncated state, ordinary-state compatibility,
nonzero fixed LoRA/control vectors, live mutation rejection and adapter identity matching,
shared physical capacity and mixed phases. Sampler continuation tests belong to the
separate generation-persistence patch.

## Tool audit

There are two existing public inference functions, `llama_encode` and
`llama_decode`. This patch adds `llama_decode_prefix` and `llama_decode_prefix_mixed`; backend graph execution,
common helpers and HTTP endpoints are callers, not additional public model
inference APIs.

- `llama-cli`, `llama-server`, common warmup/prompt processing, `llama-simple`
  and `llama-simple-chat` submit a complete prefix explicitly. Simple chat
  renders the whole retained conversation again each turn.
- `llama-bench` supports prompt processing, generation and combined runs.
  Generation-only runs prepare one seed prefix before timing. Cached depth
  runs still reject because their cached-depth/replay scheduling has no explicit
  complete-prefix boundary; library persistence itself is supported. Prompts
  must fit `-b` and `-ub`; benchmark comparisons must record attention mode.
- `llama-perplexity` and `llama-imatrix` explicitly use causal attention for
  unlabelled text. Non-causal and PrefixLM overrides reject. These measure
  causal likelihood/calibration, not conditional PrefixLM likelihood.
- Batched/parallel examples, batched-bench, lookup/lookahead, speculative
  decoding, passkey cache editing and Android incremental chat have no
  PrefixLM scheduler adaptation. The library rejects their incompatible
  context or decode operation. They are not qualified PrefixLM clients.
- Embedding/retrieval, diffusion, cvector, mtmd, debug, idle, eval-callback and
  results paths have no implicit prefix boundary. They are outside this
  supported client list; passing a model with PrefixLM metadata does not
  authorize silently treating their first ordinary decode as a prefix.
- Legacy completion uses the adapted common prompt helper, but its cache,
  interactive continuation and context-shifting paths are not qualified.
  Use CLI/server for PrefixLM chat.

The decode guard is intentional: first-call detection would misclassify a
one-token prefix, a resumed sequence or a new conversation turn. Existing
clients must declare the boundary instead of obtaining plausible but incorrect
causal output. This audit is a support inventory, not a claim that every
example has an appropriate user-facing rejection message.

## Independent reference

The existing architecture test can also consume full-forward Transformers
logits. Generate a small model with the normal converter, without a tokenizer
or model download:

```sh
python -m pip install -r requirements/requirements-convert_hf_to_gguf.txt
python -m pip install "transformers @ git+https://github.com/huggingface/transformers.git@ff2421c67f35cc83a0fbabbc2633c96734685918"
python scripts/prefix-lm-reference.py --output /tmp/prefix-reference
build/bin/test-llama-archs --prefix-reference /tmp/prefix-reference/reference.json
```

The JSON specification records model, reference files, context/batch limits,
backend selection, tolerance and output path. `n_gpu_layers` and `flash` select
GPU execution and fused attention. The report contains finiteness, maximum
absolute error, RMS error and top-token agreement per step. Full-forward
references cover bidirectional prefix visibility, causal answers, answer rollback, chunk and
single-token answers, a one-token prefix and a causal control. The reference also checks interleaved sequences and local reset/rollback. Generation uses
an explicit seed and records package versions and the GGUF hash. Re-generating
references avoids assuming random weights are identical across platforms.

The same reader accepts real-model and quantized-model references. A failure
against a strict full-precision tolerance is retained as a failure; quantized
logit drift must be assessed separately from attention semantics and language
quality. Optional `logits_dir` writes the tested rows for additional comparisons. A step may specify `rewind_to` to remove an answer suffix before decoding its tokens.

## Experimental MixedLM (opt-in)

Set `llama_context_params.mixed_lm = true` before creating a PrefixLM context.
After an initial `llama_decode_prefix`, `llama_decode_mixed_lm` retains the old
prompt KV and replaces its causal answer with a new bidirectional suffix. Pass
the previous assistant answer again, followed by the new user prompt and
assistant header, using the model's chat template. Suffix positions begin at the
previous prefix end. The whole suffix must fit `n_ubatch` and `n_batch`.

This approximates full PrefixLM: frozen older representations cannot attend to
the new suffix. It can change answer quality. The caller must check that all
frozen tokens are unchanged. Use `llama_decode_prefix` for a full refresh after
history edits or other invalidation. New answer tokens use `llama_decode`.
This option defaults false and has no server CLI integration.

Admission, shared-owner dependency checks and error isolation reuse the existing
PrefixLM path. Retained shared KV cells count once toward capacity. Exact mode
keeps snapshot version 2; MixedLM uses version 3 and rejects cross-mode restore.
Clients must rebuild against the changed context parameter structure.

`test-llama-archs --prefix-lm --arch hrm_text --seed 42` (also `--arch llama`)
checks the approximation against an independent blockwise noncausal oracle,
repeated extension, causal continuation, full refresh, opt-in rejection, invalid
positions, abort recovery, state restore and shared-cell capacity, with flash
attention both enabled and disabled. It does not establish model-quality parity.
