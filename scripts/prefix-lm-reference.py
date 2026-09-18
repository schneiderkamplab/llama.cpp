"""Generate a small Transformers PrefixLM oracle using the normal GGUF converter.

Requires the converter dependencies and Transformers commit
ff2421c67f35cc83a0fbabbc2633c96734685918. No model download is needed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import transformers
from transformers import HrmTextConfig, HrmTextForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conversion import get_model_class  # noqa: E402
import gguf  # noqa: E402

__all__ = []


class _TokenlessHrm(get_model_class("HrmTextForCausalLM")):
    model_arch = gguf.MODEL_ARCH.HRM_TEXT

    def set_vocab(self):
        self.gguf_writer.add_tokenizer_model("no_vocab")
        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(73)
    cfg = HrmTextConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                       num_layers_per_stack=1, num_attention_heads=4, num_key_value_heads=4,
                       head_dim=16, H_cycles=2, L_cycles=3, max_position_embeddings=128,
                       prefix_lm=True, embedding_scale=12.5, initializer_range=0.08,
                       attention_bias=False, mlp_bias=False, tie_word_embeddings=False)
    cfg._attn_implementation = "eager"
    model = HrmTextForCausalLM(cfg).float().eval()
    model.save_pretrained(out / "hf")
    _TokenlessHrm(out / "hf", gguf.LlamaFileType.ALL_F32, out / "model.gguf", eager=True).write()
    prefix, answer = [2, 11, 23, 37, 41, 53], [61, 73, 89]
    cases = []
    for name, prompt, continuation, causal, chunks in [
        ("prefix", prefix, [], False, []),
        ("changed-prefix", prefix[:-1] + [57], [], False, []),
        ("answer-chunk", prefix, answer, False, [3]),
        ("answer-single", prefix, answer, False, [1, 1, 1]),
        ("changed-answer", prefix, answer[:-1] + [97], False, [3]),
        ("one-token-prefix", prefix[:1], answer, False, [3]),
        ("causal-control", prefix, answer, True, [3]),
        ("other-answer", [7, 19, 31, 43, 59], [67, 79, 101], False, [3]),
    ]:
        ids = torch.tensor([prompt + continuation])
        types = torch.tensor([[0 if causal else 1] * len(prompt) + [0] * len(continuation)])
        with torch.no_grad():
            logits = model(input_ids=ids, token_type_ids=types, use_cache=False).logits[0].numpy()
        steps, pos = [], 0
        for i, count in enumerate([len(prompt)] + chunks):
            path = out / f"{name}-{i}.f32"
            logits[pos:pos + count].astype("<f4").tofile(path)
            steps.append({"tokens": ids[0, pos:pos + count].tolist(), "prefix": i == 0 and not causal,
                          "reference": str(path), "all_logits": True})
            pos += count
        cases.append({"name": name, "causal": causal, "steps": steps})
    chunk_steps = next(case["steps"] for case in cases if case["name"] == "answer-chunk")
    mixed = out / "mixed-phase.f32"
    mixed.write_bytes(b"".join(Path(step["reference"]).read_bytes() for step in chunk_steps))
    cases.append({"name": "mixed-phase", "steps": [
        {"tokens": prefix + answer, "prefix": True, "prefix_end": len(prefix),
         "reference": str(mixed), "all_logits": True}]})
    cases.append({"name": "shared-owners", "n_seq_max": 2, "kv_unified": True,
                  "steps": [{**step, "owners": [0, 1]} for step in chunk_steps]})
    suffix = out / "answer-rollback.f32"
    suffix.write_bytes(b"".join((out / f"answer-single-{i}.f32").read_bytes() for i in (2, 3)))
    cases.append({"name": "answer-rollback", "steps": [*chunk_steps,
                  {"tokens": answer[1:], "prefix": False, "rewind_to": len(prefix) + 1,
                   "reference": str(suffix)},
                  {**chunk_steps[1], "rewind_to": len(prefix)}]})
    causal_steps = next(case["steps"] for case in cases if case["name"] == "causal-control")
    causal_full = out / "causal-full.f32"
    causal_full.write_bytes(b"".join(Path(step["reference"]).read_bytes() for step in causal_steps))
    cases.append({"name": "causal-full", "causal": True, "steps": [
        {"tokens": prefix + answer, "prefix": False, "reference": str(causal_full), "all_logits": True}]})
    other_steps = next(case["steps"] for case in cases if case["name"] == "other-answer")
    cases.append({"name": "interleaved-sequences", "n_seq_max": 2, "steps": [
        chunk_steps[0], {**other_steps[0], "sequence": 1}, chunk_steps[1],
        {**other_steps[1], "sequence": 1}, chunk_steps[0],
        {**other_steps[1], "sequence": 1, "rewind_to": 5}, chunk_steps[1]]})
    spec = {"model": str(out / "model.gguf"), "n_ctx": 128, "n_ubatch": 64,
            "max_abs": 0.0001, "cases": cases, "report": str(out / "result.json"),
            "provenance": {"seed": 73, "torch": torch.__version__, "transformers": transformers.__version__,
                           "model_sha256": hashlib.sha256((out / "model.gguf").read_bytes()).hexdigest()}}
    (out / "reference.json").write_text(json.dumps(spec, indent=2) + "\n")


if __name__ == "__main__":
    _main()
