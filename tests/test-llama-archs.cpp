#include "common.h"
#include "sampling.h"
#include "log.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"
#include "nlohmann/json.hpp"

#include <fstream>
#include <cmath>
#include <algorithm>

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-arch.h"
#include "../src/llama-adapter.h"
#include "../src/llama-model.h"
#include "../src/llama-model-saver.h"

#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    size_t seed = *(const size_t *) userdata;
    std::hash<std::string> hasher;
    seed ^= hasher(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = dis(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

static void usage(char ** argv) {
    printf("Usage: %s [-a/--arch arch] [-s/--seed seed] [-o/--out dir] [-v N] [--prefix-lm] [--prefix-reference FILE] [-h/--help]\n", argv[0]);
}

static std::vector<llama_token> get_tokens(const uint32_t n_tokens, const uint32_t n_vocab, const size_t seed){
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

static gguf_context_ptr get_gguf_ctx(const llm_arch arch, const bool moe) {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());
    const uint32_t n_ctx = 256;

    uint32_t n_vocab = 128;
    uint32_t n_embd  = 256;
    uint32_t n_head  = 2;
    uint32_t n_ff    = 384;
    uint32_t n_layer = 2;
    if (arch == LLM_ARCH_LLAMA4) {
        n_layer = 4; // hparams.n_no_rope_layer_step is hard-coded to 4
    } else if (arch == LLM_ARCH_GEMMA4) {
        n_embd = 128;
        n_head = 2;
        n_ff   = 192;
        n_layer = 5; // need at least 5 for swa_pattern (every 5th is full_attention)
    } else if (arch == LLM_ARCH_GEMMA3N) {
        n_embd = 64;
        n_head = 1;
        n_ff   = 96;
        n_layer = 22; // hparams.n_layer_kv_from_start = 20 is hardcoded
    } else if (arch == LLM_ARCH_DEEPSEEK4) {
        // head size 64 so that GPU flash attention kernels support the model
        n_embd  = 512;
        n_head  = 8;
        n_ff    = 1024;
        n_layer = 4;
    } else if (arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_LAGUNA) {
        n_embd = 160; // exercise per-head tensor split granularity with head size 80
    } else if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_AFMOE) {
        n_head = 4;
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_MISTRAL4
            || arch == LLM_ARCH_HY_V4) {
        n_embd = 128;
        n_head = 1;
        n_ff   = 192;
    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_CHAMELEON) {
        n_vocab = 10240;
    } else if (arch == LLM_ARCH_QWEN3TTS) {
        //n_vocab = 4096; // must be >= the hard-coded codec head size (3072)
        n_vocab = 3072; // TODO: should be 4096, but user code cannot get `n_vocab_out` yet [TAG_LLAMA_N_VOCAB_OUT]
    } else if (arch == LLM_ARCH_HRM_TEXT) {
        n_layer = 8; // 1 layer per stack x 2 h-cycles x (3 l-cycles + 1) cache slots
    }

    uint32_t n_head_kv = n_head;
    if (arch == LLM_ARCH_QWEN3) {
        n_head_kv = 1; // MQA coverage
    } else if (arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_AFMOE) {
        n_head_kv = 2; // GQA coverage
    }
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));

    if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        std::vector<uint32_t> n_ff_per_layer;
        n_ff_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_ff_per_layer.push_back(il <= 1 ? 0 : n_ff);
        }
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff_per_layer);
    } else {
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
    }

    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,   false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,             1.0f);
    ms.add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,      uint32_t(64));
    ms.add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,    uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));

    if (arch == LLM_ARCH_PLAMO2 || arch == LLM_ARCH_JAMBA || arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE ||
            arch == LLM_ARCH_GRANITE_HYBRID || arch == LLM_ARCH_LFM2 || arch == LLM_ARCH_LFM2MOE || arch == LLM_ARCH_KIMI_LINEAR ||
            arch == LLM_ARCH_BAILINGMOE3 || arch == LLM_ARCH_KIMI_K3) {
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_per_layer;
        n_head_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_per_layer.push_back(il == 1 ? 0 : n_head);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head_per_layer);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_per_layer);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(1) : n_head_kv);
    }

    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, 8.0f);
    if (arch == LLM_ARCH_DEEPSEEK4) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,   n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH, n_embd_head);
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,   n_embd_head/2);
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_MISTRAL4
            || arch == LLM_ARCH_HY_V4) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       uint32_t(576));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
        if (arch == LLM_ARCH_DOTS3NOTE) {
            // SWA layers reuse the same MLA geometry as the full layers in this fixture
            ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,       uint32_t(576));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA_SWA,   uint32_t(192));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA_SWA, uint32_t(128));
            ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,             10000.0f);
            // indexer on the full-attention layers (inverse of the swa pattern)
            std::vector<uint32_t> indexer_types;
            indexer_types.reserve(n_layer);
            for (uint32_t il = 0; il < n_layer; il++) {
                indexer_types.push_back(il % 2 ? 0 : 1);
            }
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
        }
    } else if (arch == LLM_ARCH_MINIMAX_M3) {
        // partial rotary: n_rot must not exceed the indexer key length (64)
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
    }
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(64) : uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);

    if (arch == LLM_ARCH_GEMMA4) {
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,      n_embd/2);
        ms.add_kv(LLM_KV_ATTENTION_SHARED_KV_LAYERS,      uint32_t(0));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,        n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,      n_embd_head);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,              10000.0f);
        // SWA pattern: every 5th layer is full attention (matches E2B layer_types)
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(5));
    } else if (arch == LLM_ARCH_COHERE2MOE || arch == LLM_ARCH_MIMO2 || arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_SPARK2_5 ||
            arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_GRANITE_SWA || arch == LLM_ARCH_DOTS3NOTE ||
            arch == LLM_ARCH_MAPLE) {
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back(il % 2);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    }

    // MSA requires one indexer head per GQA (KV) head, unlike the DSA archs where the
    // indexer head count is independent of the main attention head count.
    if (arch == LLM_ARCH_QWEN4EXP) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,    uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK, uint32_t(8));
        // without this the QSA layers fall back to dense and go uncovered
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS, std::vector<uint32_t>(n_layer, 4));

        // has_cell_ext() needs ple_n_heads here: the indexer cache serializes no ext without it
        const uint32_t ple_ngram_size      = 3;
        const uint32_t ple_heads_per_ngram = 2;
        const uint32_t ple_n_heads         = (ple_ngram_size - 1)*ple_heads_per_ngram;
        GGML_ASSERT(n_embd % ple_n_heads == 0);
        const uint32_t ple_head_dim = n_embd/ple_n_heads;

        std::vector<uint64_t> ple_head_offsets(ple_n_heads);
        std::vector<uint64_t> ple_head_vocab_sizes(ple_n_heads, n_vocab);
        for (uint32_t h = 0; h < ple_n_heads; h++) {
            ple_head_offsets[h] = uint64_t(h)*n_vocab;
        }

        // the PLE history lives in the recurrent cache, so it must sit on a linear attention layer
        ms.add_kv(LLM_KV_PLE_LAYERS,                  std::vector<uint32_t>({ 0 }));
        ms.add_kv(LLM_KV_PLE_NGRAM_SIZE,              ple_ngram_size);
        ms.add_kv(LLM_KV_PLE_HEADS_PER_NGRAM,         ple_heads_per_ngram);
        ms.add_kv(LLM_KV_PLE_CONV_KERNEL,             uint32_t(4));
        ms.add_kv(LLM_KV_PLE_EOS_TOKEN_ID,            uint32_t(0));
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  ple_head_dim);
        ms.add_kv(LLM_KV_PLE_LAYER_MULTIPLIERS,       std::vector<uint64_t>({ 1, 3, 5 }));
        ms.add_kv(LLM_KV_PLE_HEAD_OFFSETS,            ple_head_offsets);
        ms.add_kv(LLM_KV_PLE_HEAD_VOCAB_SIZES,        ple_head_vocab_sizes);
    }

    // minimax-m3 keeps one indexer head per GQA head; the rest use a fixed 64 to match the fused
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,   arch == LLM_ARCH_MINIMAX_M3 ? n_head : uint32_t(64));
    // qwen4exp ropes indexer keys with the main rotary width, so its head can't be < n_rot
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,
              arch == LLM_ARCH_QWEN4EXP ? n_embd_head : uint32_t(128));

    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,   uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, uint32_t(1));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));

    if (arch == LLM_ARCH_HY_V4) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,     uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,   1.0e-6f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_MAGNITUDE, 2.0f);
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,           10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,       1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,        true);
        // layer 0 must own an indexer, the odd layers share it
        std::vector<uint32_t> indexer_types;
        indexer_types.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            indexer_types.push_back(il % 2 ? 0 : 1);
        }
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
    }

    if (arch == LLM_ARCH_DEEPSEEK4) {
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,         uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,           uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,            std::vector<uint32_t>({0, 0, 4, 128}));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,    160000.0f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,               uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,             1.0e-6f);
        ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                      uint32_t(0));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,                      10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                  1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                   true);
    }

    if (arch == LLM_ARCH_HRM_TEXT) {
        // 8 cache slots alias 2 physical blocks: 1 low-stack layer + 1 high-stack layer
        ms.add_kv(LLM_KV_HRM_LAYERS_PER_STACK, uint32_t(1));
        ms.add_kv(LLM_KV_HRM_H_CYCLES,         uint32_t(2));
        ms.add_kv(LLM_KV_HRM_L_CYCLES,         uint32_t(3));
    }

    if (arch == LLM_ARCH_MAPLE) {
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP, 7.0f);
    }

    ms.add_kv(LLM_KV_TOKENIZER_MODEL,         "no_vocab");
    // ms.add_kv(LLM_KV_DENSE_2_FEAT_OUT,     n_embd);
    // ms.add_kv(LLM_KV_DENSE_3_FEAT_IN,      n_embd);

    if (moe) {
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
        ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, n_ff / 2);  // distinct from n_ff so a saver key-clobber surfaces on reload
        ms.add_kv(LLM_KV_EXPERT_LATENT_LENGTH,       n_ff);
        ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,  uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_COUNT,               uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,         arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(4) : uint32_t(2)); // sqrtsoftplus : sigmoid
        ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,         1.0f);
        ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,          uint32_t(1));
    }

    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-7f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_QWEN4EXP ? 256 : 2*n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           arch == LLM_ARCH_PLAMO2 ? 0 : uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_KDA_SAFE_GATE,              true);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,       -5.0f);
    if (arch == LLM_ARCH_BAILINGMOE3) {
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,   std::vector<float>({0.0f, 4.0f}));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_SHEXP, std::vector<float>({0.0f, 5.0f}));
    }
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,             n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,         uint32_t(3));
    ms.add_kv(LLM_KV_RESIDUAL_SCALE,            3.5565588200778455f);
    ms.add_kv(LLM_KV_ATTN_RES_BLOCK_SIZE,       uint32_t(12));
    ms.add_kv(LLM_KV_ACTIVATION_SITU_BETA,      4.0f);
    ms.add_kv(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, 25.0f);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,      -5.0f);

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv1.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv2.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "convnext.%" PRIu32 ".dw.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return ret;
}

static bool silent_model_load_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

static std::pair<llama_model_ptr, llama_context_ptr> get_model_and_ctx(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER, bool encode = false) {
    GGML_ASSERT((gguf_ctx == nullptr) != (file == nullptr));
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    std::vector<ggml_backend_dev_t> devs_copy = devs;
    devs_copy.push_back(nullptr);
    model_params.devices = devs_copy.data();
    model_params.split_mode = split_mode;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    if (!encode) {
        ctx_params.n_ubatch = 64;
    }

    size_t tmp = seed;
    llama_model_ptr model(gguf_ctx != nullptr ?
        llama_model_init_from_user(gguf_ctx, set_tensor_data, &tmp, model_params) :
        llama_model_load_from_file_ptr(file, model_params));
    if (!model) {
        throw std::runtime_error("failed to create llama model");
    }
    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }
    return std::make_pair(std::move(model), std::move(lctx));
}

static std::vector<float> get_logits(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, bool encode = false) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_ctx    = llama_n_ctx(lctx);
    const uint32_t n_tokens = tokens.size();
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    GGML_ASSERT(n_tokens <= n_ctx);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        common_batch_add(batch, tokens[pos], pos, {0}, true);
    }
    batch.n_tokens = n_tokens;
    if (encode) {
        if (llama_encode(lctx, batch)) {
            llama_batch_free(batch);
            throw std::runtime_error("failed to encode batch");
        }
    }
    if (llama_decode(lctx, batch)) {
        llama_batch_free(batch);
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> ret;
    ret.reserve(n_tokens*n_vocab);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            ret.push_back(logits_ith[j]);
        }
    }
    llama_batch_free(batch);
    return ret;
}

static bool moe_mandatory(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_COHERE2MOE:
        case LLM_ARCH_GROK:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_QWEN4EXP:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_DEEPSEEK32:
        case LLM_ARCH_DOTS3NOTE:
        case LLM_ARCH_DEEPSEEK4:
        case LLM_ARCH_GLM4_MOE:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_EXAONE_MOE:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_BAILINGMOE3:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_AFMOE:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_HY_V3:
        case LLM_ARCH_HY_V4:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_LFM2MOE:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_GROVEMOE:
        case LLM_ARCH_MINIMAX_01:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_MINIMAX_M3:
        case LLM_ARCH_RND1:
        case LLM_ARCH_PADDLEOCR:
        case LLM_ARCH_MIMO2:
        case LLM_ARCH_KIMI_LINEAR:
        case LLM_ARCH_KIMI_K3:
        case LLM_ARCH_STEP35:
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_MELLUM:
        case LLM_ARCH_LAGUNA:
        case LLM_ARCH_MAPLE:
            return true;
        default:
            return false;
    }
}

static bool moe_implemented(const llm_arch arch) {
    if (moe_mandatory(arch)) {
        return true;
    }
    switch (arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_LLAMA_EMBED:
            return true;
        default:
            return false;
    }
}

static bool arch_supported(const llm_arch arch) {
    if (arch == LLM_ARCH_CLIP || arch == LLM_ARCH_GPTJ || arch == LLM_ARCH_UNKNOWN) {
        return false; // These models don't have usable implementations.
    }
    if (arch == LLM_ARCH_CHAMELEON) {
        return false; // Only half-implemented and to be removed in the future.
    }
    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        return false; // FIXME CUDA backend crashes.
    }
    if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        return false; // FIXME @ngxson
    }
    if (arch == LLM_ARCH_GRANITE_SWITCH) {
        return false; // FIXME adapter fixture
    }
    if (arch == LLM_ARCH_LLAMA_EMBED || arch == LLM_ARCH_GEMMA_EMBEDDING || arch == LLM_ARCH_T5ENCODER) {
        return false; // FIXME Embedding (?) models produce inconsistent results.
    }
    if (arch == LLM_ARCH_RWKV6 || arch == LLM_ARCH_RWKV6QWEN2 || arch == LLM_ARCH_RWKV7 || arch == LLM_ARCH_ARWKV7) {
        return false; // FIXME RWKV models hang indefinitely.
    }
    if (arch == LLM_ARCH_BERT || arch == LLM_ARCH_MODERN_BERT || arch == LLM_ARCH_NOMIC_BERT || arch == LLM_ARCH_NOMIC_BERT_MOE ||
            arch == LLM_ARCH_NEO_BERT || arch == LLM_ARCH_JINA_BERT_V2 || arch == LLM_ARCH_JINA_BERT_V3 || arch == LLM_ARCH_EUROBERT) {
        return false; // TODO vocab
    }
    if (arch == LLM_ARCH_PLM) {
        return false; // TODO tensor shapes
    }
    if (arch == LLM_ARCH_DEEPSEEK2OCR) {
        return false;
    }
    // FIXME: these hit scheduler/view-backed-output issues with WebGPU on CI.
#ifdef GGML_USE_WEBGPU
    if (arch == LLM_ARCH_DEEPSEEK32 || arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_DOTS3NOTE || arch == LLM_ARCH_QWEN4EXP ||
            arch == LLM_ARCH_HY_V4) {
        return false;
    }
#endif // GGML_USE_WEBGPU

    // FIXME: jamba produces incorrect output (~0.55 NMSE vs CPU) on the HIP
    // backend on RDNA3.5 (gfx1151); the SSM kernels need investigation.
#ifdef GGML_USE_HIP
    if (arch == LLM_ARCH_JAMBA) {
        return false;
    }
#endif // GGML_USE_HIP

    return true;
}

static int save_models(const llm_arch target_arch, const size_t seed, const int verbosity, const std::string & dir) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (target_arch != LLM_ARCH_UNKNOWN && arch != target_arch) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue; // FIXME: ISWA KV cache initialization needs more fixture params
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH) {
            continue;
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            if (!llama_model_saver_supports_arch(arch) || !arch_supported(arch)) {
                LOG_INF("%s: %s model (%s) is unsupported, skipping\n", __func__, llm_arch_name(arch), moe ? "MoE" : "dense");
                continue;
            }
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            auto model_and_ctx = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {});
            const std::string path = dir + "/" + llm_arch_name(arch) + (moe ? "-moe.gguf" : "-dense.gguf");
            LOG_INF("%s: Saving %s model (%s) to %s...\n", __func__, llm_arch_name(arch), moe ? "MoE" : "dense", path.c_str());
            llama_model_save_to_file(model_and_ctx.first.get(), path.c_str());
        }
    }
    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return 0;
}

static int test_backends(const llm_arch target_arch, const size_t seed, const int verbosity) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    const std::vector<llama_token> tokens = get_tokens(128, 128, seed);

    struct device_config {
        std::vector<ggml_backend_dev_t> devs;
        std::string                     label;
        llama_split_mode                split_mode;

        device_config(std::vector<ggml_backend_dev_t> devs, std::string name, llama_split_mode split_mode)
            : devs(std::move(devs)), label(std::move(name)), split_mode(split_mode) {}
    };

    std::vector<device_config> dev_configs;
    size_t max_device_label_length = 4;
    {
        std::vector<ggml_backend_dev_t> devices_meta;
        {
            const size_t device_count = ggml_backend_dev_count();
            for (size_t i = 0; i < device_count; i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{dev}, ggml_backend_dev_description(dev), LLAMA_SPLIT_MODE_LAYER);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());

                // cpu-based devices cannot be used in tensor split mode
                if (ggml_backend_dev_buffer_type(dev) != ggml_backend_cpu_buffer_type()) {
                    devices_meta.push_back(dev);
                }
            }
        }

        dev_configs.emplace_back(devices_meta, "Meta", LLAMA_SPLIT_MODE_TENSOR);
    }

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llm_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llm_arch_name(arch)));
    }

    const std::string template_header  = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|%15s|%9s|\n";
    const std::string template_row_cfg = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|";
    const std::string template_row_res = "%15s %10s|%20s|\n";

    bool all_ok = true;
    common_log_flush(common_log_main());
    printf(template_header.c_str(), "Model arch.", "Device", "Config", "NMSE vs. CPU", "Roundtrip");
    printf("|");
    for (size_t i = 0; i < max_arch_name_length; i++) {
        printf("-");
    }
    printf("|");
    for (size_t i = 0; i < max_device_label_length; i++) {
        printf("-");
    }
    printf("|------|---------------|---------|\n");
    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (target_arch != LLM_ARCH_UNKNOWN && arch != target_arch) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue; // FIXME: ISWA KV cache initialization needs more fixture params
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH) {
            continue;
        }

        const bool encode = arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE || arch == LLM_ARCH_RND1;
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const std::string config_name = moe ? "MoE" : "Dense";
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            if (arch == LLM_ARCH_BAILINGMOE3) {
                GGML_ASSERT(gguf_remove_key(gguf_ctx.get(), "bailingmoe3.kda.safe_gate") >= 0);
            }
            std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_cpu;
            std::vector<float> logits_cpu;
            for (device_config & dc : dev_configs) {
                // print test config first; should anything fail during model loading or inference, at least we know which test case caused it
                printf(template_row_cfg.c_str(),
                    llm_arch_name(arch), dc.label.c_str(), config_name.c_str());
                fflush(stdout);

                std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_dev;
                std::vector<float> logits_dev;
                std::string status_nmse      = "\033[1;33mSKIP\033[0m";
                std::string status_roundtrip = "\033[1;33mSKIP\033[0m";
                char nmse_str[12] = {0};

                bool skip = !arch_supported(arch) || (dc.split_mode == LLAMA_SPLIT_MODE_TENSOR && dc.devs.empty());
                if (!skip) {
                    if (logits_cpu.empty()) {
                        model_and_ctx_cpu = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, encode);
                        logits_cpu = get_logits(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, encode);
                    }
                    if (dc.split_mode != LLAMA_SPLIT_MODE_TENSOR || llm_arch_supports_sm_tensor(arch)) {
                        model_and_ctx_dev = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, dc.devs, dc.split_mode, encode);
                        logits_dev = get_logits(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens, encode);
                        const double nmse_val = nmse(logits_cpu, logits_dev);
                        snprintf(nmse_str, sizeof(nmse_str), "(%.2e)", nmse_val);
                        status_nmse = "\033[1;32mOK\033[0m";
                        if (nmse_val > 1e-4) {
                            all_ok = false;
                            status_nmse = "\033[1;31mFAIL\033[0m";
                        }
                    }

                    FILE * file = tmpfile(); // Can be null on Windows without administrator privileges.
                    // FIXME: when adding a tensor to a gguf_context a copy is made, this changes the pointer which the meta backend
                    //     in turn uses to map the tensors to their simple equivalents - this is fundamentally incompatible
                    if (file != nullptr && llama_model_saver_supports_arch(arch) && dc.split_mode != LLAMA_SPLIT_MODE_TENSOR) {
                        GGML_ASSERT(model_and_ctx_dev.first && model_and_ctx_dev.second);
                        llama_model_saver ms = llama_model_saver(model_and_ctx_dev.first.get());
                        ms.add_kv_from_model();
                        ms.add_tensors_from_model();
                        ms.save(file);
                        rewind(file);

                        auto model_and_ctx_roundtrip = get_model_and_ctx(nullptr, file, seed, dc.devs, dc.split_mode, encode);
                        const std::vector<float> logits_roundtrip = get_logits(
                            model_and_ctx_roundtrip.first.get(), model_and_ctx_roundtrip.second.get(), tokens, encode);
                        status_roundtrip = "\033[1;32mOK\033[0m";
                        GGML_ASSERT(logits_roundtrip.size() == logits_dev.size());
                        for (size_t i = 0; i < logits_roundtrip.size(); i++) {
                            if (logits_roundtrip[i] != logits_dev[i]) {
                                all_ok = false;
                                status_roundtrip = "\033[1;31mFAIL\033[0m";
                                break;
                            }
                        }
                    }
                }

                // log the results for this test case
                printf(template_row_res.c_str(),
                    status_nmse.c_str(), nmse_str, status_roundtrip.c_str());
            }
        }
    }
    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return all_ok ? 0 : 1;
}

static int test_prefix_reference(const std::string & path) {
    using json = nlohmann::ordered_json;
    std::ifstream input(path);
    const auto spec = json::parse(input);
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = spec.value("n_gpu_layers", 0);
    llama_model_ptr model(llama_model_load_from_file(spec.at("model").get<std::string>().c_str(), mp));
    if (!model) { throw std::runtime_error("reference model load failed"); }
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    json results = json::array();
    bool passed = true;
    for (const auto & item : spec.at("cases")) {
        auto cp = llama_context_default_params();
        cp.n_ctx = spec.value("n_ctx", 128);
        cp.n_seq_max = item.value("n_seq_max", 1);
        cp.kv_unified = item.value("kv_unified", false);
        cp.n_batch = cp.n_ctx;
        cp.n_ubatch = spec.value("n_ubatch", 64);
        cp.n_threads = cp.n_threads_batch = 4;
        cp.type_k = cp.type_v = GGML_TYPE_F32;
        cp.flash_attn_type = spec.value("flash", false) ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.attention_type = item.value("causal", false) ? LLAMA_ATTENTION_TYPE_CAUSAL : LLAMA_ATTENTION_TYPE_PREFIX_LM;
        llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
        if (!ctx) { throw std::runtime_error("reference context init failed"); }
        std::map<llama_seq_id, llama_pos> positions;
        for (const auto & step : item.at("steps")) {
            const auto owners = step.value("owners", std::vector<llama_seq_id>{step.value("sequence", 0)});
            const auto seq = owners.at(0);
            auto & pos = positions[seq];
            if (step.at("prefix").get<bool>()) { pos = 0; }
            if (step.contains("rewind_to")) {
                const auto target = step.at("rewind_to").get<llama_pos>();
                if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), seq, target, -1)) {
                    throw std::runtime_error("reference answer rollback failed");
                }
                pos = target;
            }
            const auto tokens = step.at("tokens").get<std::vector<llama_token>>();
            const bool all = step.value("all_logits", true);
            auto batch = llama_batch_init(tokens.size(), 0, owners.size());
            for (size_t i = 0; i < tokens.size(); ++i) {
                common_batch_add(batch, tokens[i], pos + i, owners, all || i + 1 == tokens.size());
            }
            const llama_seq_id sequence = seq;
            const llama_pos prefix_end = step.value("prefix_end", 0);
            const int ret = prefix_end > 0
                ? llama_decode_prefix_mixed(ctx.get(), batch, &sequence, &prefix_end, 1)
                : step.at("prefix").get<bool>() ? llama_decode_prefix(ctx.get(), batch) : llama_decode(ctx.get(), batch);
            llama_batch_free(batch);
            if (ret != 0) { throw std::runtime_error("reference decode failed"); }
            const size_t rows = all ? tokens.size() : 1;
            std::vector<float> expected(rows * nv);
            std::ifstream ref(step.at("reference").get<std::string>(), std::ios::binary);
            ref.read(reinterpret_cast<char *>(expected.data()), expected.size() * sizeof(float));
            if (!ref || ref.peek() != EOF) { throw std::runtime_error("invalid reference logit size"); }
            std::ofstream dump;
            if (spec.contains("logits_dir")) {
                dump.open(spec.at("logits_dir").get<std::string>() + "/" + item.at("name").get<std::string>() + "-" + std::to_string(pos) + ".f32", std::ios::binary);
                if (!dump) { throw std::runtime_error("cannot write logits"); }
            }
            double max_abs = 0, squared = 0;
            int same_top = 0;
            bool finite = true;
            for (size_t i = 0; i < rows; ++i) {
                const float * got = llama_get_logits_ith(ctx.get(), all ? i : tokens.size() - 1);
                if (dump.is_open()) { dump.write(reinterpret_cast<const char *>(got), nv * sizeof(float)); }
                const float * want = expected.data() + i * nv;
                same_top += std::max_element(got, got + nv) - got == std::max_element(want, want + nv) - want;
                for (int j = 0; j < nv; ++j) {
                    finite &= std::isfinite(got[j]) && std::isfinite(want[j]);
                    const double delta = double(got[j]) - want[j];
                    max_abs = std::max(max_abs, std::abs(delta));
                    squared += delta * delta;
                }
            }
            if (dump.is_open() && !dump) { throw std::runtime_error("logit write failed"); }
            const bool ok = finite && max_abs <= spec.value("max_abs", 0.0001);
            passed &= ok;
            results.push_back({{"case", item.at("name")}, {"sequence", seq}, {"position", pos}, {"rows", rows},
                               {"finite", finite}, {"max_abs", max_abs}, {"rmse", std::sqrt(squared / expected.size())},
                               {"top1_matches", same_top}, {"pass", ok}});
            pos += tokens.size();
            for (auto owner : owners) { positions[owner] = pos; }
        }
    }
    json report = {{"pass", passed}, {"specification", spec}, {"steps", results}};
    std::ofstream output(spec.at("report").get<std::string>());
    output << report.dump(2) << '\n';
    if (!output) { throw std::runtime_error("cannot write reference result"); }
    printf("independent reference: %zu steps, pass=%d\n", results.size(), passed);
    return passed ? 0 : 1;
}

static int test_prefix_lm(const llm_arch arch, size_t seed) {
    if (arch != LLM_ARCH_HRM_TEXT && arch != LLM_ARCH_LLAMA) {
        throw std::runtime_error("--prefix-lm requires --arch hrm_text or llama");
    }
    int checks = 0;
    auto require = [&](bool ok, const char * message) {
        if (!ok) { throw std::runtime_error(message); }
        ++checks;
    };
    std::vector<std::vector<ggml_backend_dev_t>> devices = {{}};
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * device = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU) { devices.push_back({device}); }
    }
    for (const auto & devs : devices) {
        auto gguf = get_gguf_ctx(arch, false);
        if (arch == LLM_ARCH_HRM_TEXT) {
            llama_model_saver(arch, gguf.get()).add_kv(LLM_KV_HRM_PREFIX_LM, true);
        }
        auto owner = get_model_and_ctx(gguf.get(), nullptr, seed, devs);
        auto * model = owner.first.get();
        require(llama_get_attention_type(owner.second.get()) ==
                (arch == LLM_ARCH_HRM_TEXT ? LLAMA_ATTENTION_TYPE_PREFIX_LM : LLAMA_ATTENTION_TYPE_CAUSAL), "model default attention");
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        {
            auto cp = llama_context_default_params();
            cp.attention_type = LLAMA_ATTENTION_TYPE_PREFIX_LM;
            cp.kv_unified = true; cp.n_seq_max = 2; cp.n_ctx = cp.n_batch = cp.n_ubatch = 16;
            llama_context_ptr shared(llama_init_from_model(model, cp));
            require(bool(shared), "shared capacity context");
            auto * memory = llama_get_memory(shared.get());
            auto tokens = get_tokens(12, n_vocab, seed);
            require(llama_decode_prefix(shared.get(), llama_batch_get_one(tokens.data(), tokens.size())) == 0, "shared capacity prefix");
            llama_memory_seq_cp(memory, 0, 7, 0, -1);
            require(llama_memory_seq_pos_max(memory, 7) == 11, "fork counts physical cells once");
            auto batch = llama_batch_init(4, 0, 1);
            for (int i = 0; i < 2; ++i) {
                common_batch_add(batch, tokens[i], 12 + i, {0}, true);
                common_batch_add(batch, tokens[i + 2], 12 + i, {7}, true);
            }
            require(llama_decode(shared.get(), batch) == 0, "divergent answers fill remaining physical capacity");
            common_batch_clear(batch);
            common_batch_add(batch, tokens[0], 14, {0}, true);
            require(llama_decode(shared.get(), batch) == -1, "physical capacity enforced before padded capacity");
            std::vector<uint8_t> state(llama_state_get_size(shared.get()));
            require(llama_state_get_data(shared.get(), state.data(), state.size()) == state.size(), "save shared physical capacity");
            llama_memory_clear(memory, false);
            require(llama_state_set_data(shared.get(), state.data(), state.size()) == state.size(), "restore shared capacity without double counting");
            auto replacement = get_tokens(5, n_vocab, seed + 1);
            require(llama_decode_prefix(shared.get(), llama_batch_get_one(replacement.data(), replacement.size())) == -1,
                    "replacement cannot reclaim sibling-owned shared cells");
            llama_batch_free(batch);
            llama_memory_clear(memory, false);
            batch = llama_batch_init(12, 0, 2);
            for (int i = 0; i < 12; ++i) { common_batch_add(batch, tokens[i], i, {0, 7}, true); }
            require(llama_decode_prefix(shared.get(), batch) == 0, "shared input prefix counts physical rows once");
            common_batch_clear(batch);
            for (int i = 0; i < 4; ++i) { common_batch_add(batch, tokens[i], 12+i, {0, 7}, true); }
            require(llama_decode(shared.get(), batch) == 0, "shared answers fill physical capacity once");
            common_batch_clear(batch);
            common_batch_add(batch, tokens[0], 16, {0, 7}, true);
            require(llama_decode(shared.get(), batch) == -1, "shared input cannot exceed physical capacity");
            llama_batch_free(batch);
        }
        for (bool flash : {false, true}) {
            auto params = llama_context_default_params();
            params.n_ctx = 128;
            params.n_batch = 128;
            params.n_ubatch = 64;
            params.n_threads = params.n_threads_batch = 4;
            params.type_k = params.type_v = GGML_TYPE_F32;
            params.flash_attn_type = flash ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
            params.attention_type = LLAMA_ATTENTION_TYPE_PREFIX_LM;
            llama_context_ptr ctx(llama_init_from_model(model, params));
            require(bool(ctx), "create explicit PrefixLM context");
            require(llama_get_attention_type(ctx.get()) == LLAMA_ATTENTION_TYPE_PREFIX_LM, "selected mode");
            auto * mem = llama_get_memory(ctx.get());
            auto input = get_tokens(8, n_vocab, seed);
            auto answer = get_tokens(4, n_vocab, seed + 1);
            auto run = [&](llama_context * target, const std::vector<llama_token> & tokens, int pos, bool prefix, llama_seq_id seq = 0, bool mixed = false) {
                auto batch = llama_batch_init(tokens.size(), 0, 1);
                for (size_t i = 0; i < tokens.size(); ++i) { common_batch_add(batch, tokens[i], pos + i, {seq}, true); }
                const int code = mixed ? llama_decode_mixed_lm(target, batch) :
                    prefix ? llama_decode_prefix(target, batch) : llama_decode(target, batch);
                llama_batch_free(batch);
                require(code == 0, "decode");
                std::vector<float> result;
                for (size_t i = 0; i < tokens.size(); ++i) {
                    const auto * logits = llama_get_logits_ith(target, i);
                    result.insert(result.end(), logits, logits + n_vocab);
                }
                return result;
            };
            auto equal = [&](const std::vector<float> & a, const std::vector<float> & b) {
                require(nmse(a, b) < 1e-6, "logit parity");
            };
            require(llama_decode(ctx.get(), llama_batch_get_one(input.data(), input.size())) == -1, "answer without prefix rejected");
            // Independent oracle: freeze older KV, use noncausal attention only for each new block.
            {
                require(!params.mixed_lm, "MixedLM defaults off");
                require(llama_decode_mixed_lm(ctx.get(), llama_batch_get_one(input.data(), input.size())) == -1,
                        "MixedLM requires explicit opt-in");
                auto mp = params;
                mp.mixed_lm = true;
                llama_context_ptr mixed(llama_init_from_model(model, mp));
                require(bool(mixed), "create MixedLM context");
                require(llama_decode_mixed_lm(mixed.get(), llama_batch_get_one(input.data(), input.size())) == -1,
                        "MixedLM requires an existing prefix");
                auto rp = params;
                rp.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;
                llama_context_ptr oracle(llama_init_from_model(model, rp));
                require(bool(oracle), "create block attention oracle");
                rp.mixed_lm = true;
                llama_context_ptr invalid(llama_init_from_model(model, rp));
                require(!invalid, "MixedLM rejects ordinary causal contexts");
                llama_set_causal_attn(oracle.get(), false);
                equal(run(oracle.get(), input, 0, false), run(mixed.get(), input, 0, true));
                llama_set_causal_attn(oracle.get(), true);
                equal(run(oracle.get(), answer, input.size(), false), run(mixed.get(), answer, input.size(), false));
                auto suffix = answer;
                auto user = get_tokens(5, n_vocab, seed + 9);
                suffix.insert(suffix.end(), user.begin(), user.end());
                require(llama_memory_seq_rm(llama_get_memory(oracle.get()), 0, input.size(), -1), "trim oracle answer");
                llama_set_causal_attn(oracle.get(), false);
                equal(run(oracle.get(), suffix, input.size(), false), run(mixed.get(), suffix, input.size(), false, 0, true));
                const int end = input.size() + suffix.size();
                auto bad = llama_batch_init(1, 0, 1);
                common_batch_add(bad, answer[0], end + 1, {0}, true);
                require(llama_decode_mixed_lm(mixed.get(), bad) == -1, "MixedLM rejects noncontiguous suffix");
                llama_batch_free(bad);
                require(llama_memory_seq_pos_max(llama_get_memory(mixed.get()), 0) == end - 1, "invalid suffix preserves KV");
                std::vector<uint8_t> snapshot(llama_state_get_size(mixed.get()));
                require(llama_state_get_data(mixed.get(), snapshot.data(), snapshot.size()) == snapshot.size(), "save MixedLM state");
                require(llama_state_set_data(ctx.get(), snapshot.data(), snapshot.size()) == 0, "exact context rejects approximate state");
                auto third = get_tokens(3, n_vocab, seed + 10);
                auto expected = run(oracle.get(), third, end, false);
                equal(expected, run(mixed.get(), third, end, false, 0, true));
                llama_memory_clear(llama_get_memory(mixed.get()), false);
                require(llama_state_set_data(mixed.get(), snapshot.data(), snapshot.size()) == snapshot.size(), "restore MixedLM state");
                equal(expected, run(mixed.get(), third, end, false, 0, true));
                llama_set_causal_attn(oracle.get(), true);
                equal(run(oracle.get(), answer, end + third.size(), false), run(mixed.get(), answer, end + third.size(), false));
                run(ctx.get(), input, 0, true);
                std::vector<uint8_t> exact(llama_state_get_size(ctx.get()));
                require(llama_state_get_data(ctx.get(), exact.data(), exact.size()) == exact.size(), "save exact prefix");
                require(llama_state_set_data(mixed.get(), exact.data(), exact.size()) == 0, "MixedLM rejects incompatible exact envelope");
                equal(run(ctx.get(), input, 0, true), run(mixed.get(), input, 0, true));
                bool abort = true;
                llama_set_abort_callback(mixed.get(), [](void * data) { return *static_cast<bool *>(data); }, &abort);
                require(llama_decode_mixed_lm(mixed.get(), llama_batch_get_one(suffix.data(), suffix.size())) != 0, "MixedLM abort");
                require(llama_memory_seq_pos_max(llama_get_memory(mixed.get()), 0) == -1, "MixedLM abort clears affected KV");
                abort = false;
                equal(run(ctx.get(), input, 0, true), run(mixed.get(), input, 0, true));
                // Shared frozen cells and shared new rows each consume capacity once.
                mp.n_ctx = 16; mp.n_seq_max = 2; mp.kv_unified = true;
                llama_context_ptr shared(llama_init_from_model(model, mp));
                require(bool(shared), "create shared MixedLM context");
                auto batch = llama_batch_init(8, 0, 2);
                for (int i = 0; i < 8; ++i) { common_batch_add(batch, input[i], i, {0, 7}, true); }
                require(llama_decode_prefix(shared.get(), batch) == 0, "shared MixedLM initial prefix");
                common_batch_clear(batch);
                for (int i = 0; i < 4; ++i) { common_batch_add(batch, answer[i], 8 + i, {0, 7}, true); }
                require(llama_decode(shared.get(), batch) == 0, "shared MixedLM causal answer");
                require(llama_decode_mixed_lm(shared.get(), batch) == 0, "shared MixedLM capacity counts retained KV once");
                require(llama_memory_seq_pos_max(llama_get_memory(shared.get()), 7) == 11, "shared suffix positions");
                llama_batch_free(batch);
                llama_memory_clear(mem, false);
            }
            auto reference = run(ctx.get(), input, 0, true);
            auto chunk = run(ctx.get(), answer, input.size(), false);
            run(ctx.get(), input, 0, true);
            std::vector<float> singles;
            for (size_t i = 0; i < answer.size(); ++i) {
                auto logits = run(ctx.get(), {answer[i]}, input.size() + i, false);
                singles.insert(singles.end(), logits.begin(), logits.end());
            }
            equal(chunk, singles);
            auto long_answer = get_tokens(65, n_vocab, seed + 2);
            run(ctx.get(), input, 0, true);
            auto split_internal = run(ctx.get(), long_answer, input.size(), false);
            run(ctx.get(), input, 0, true);
            auto split_explicit = run(ctx.get(), std::vector<llama_token>(long_answer.begin(), long_answer.begin() + 32), input.size(), false);
            auto tail = run(ctx.get(), std::vector<llama_token>(long_answer.begin() + 32, long_answer.end()), input.size() + 32, false);
            split_explicit.insert(split_explicit.end(), tail.begin(), tail.end());
            equal(split_internal, split_explicit);
            auto changed_answer = answer;
            changed_answer.back() = (changed_answer.back() + 1) % n_vocab;
            run(ctx.get(), input, 0, true);
            auto changed = run(ctx.get(), changed_answer, input.size(), false);
            equal(std::vector<float>(chunk.begin(), chunk.end() - n_vocab), std::vector<float>(changed.begin(), changed.end() - n_vocab));
            auto changed_input = input;
            changed_input.back() = (changed_input.back() + 1) % n_vocab;
            auto future = run(ctx.get(), changed_input, 0, true);
            const double visibility = nmse(std::vector<float>(reference.begin(), reference.begin() + n_vocab),
                                           std::vector<float>(future.begin(), future.begin() + n_vocab));
            printf("prefix visibility nmse = %.12g\n", visibility);
            require(visibility > 0, "prefix future visibility");
            equal(reference, run(ctx.get(), input, 0, true));
            {
                llama_adapter_lora adapter(model);
                adapter.alpha = 2.0f;
                const auto * output = model->get_tensor("output.weight");
                require(output != nullptr, "output tensor for adapter fixture");
                ggml_context_ptr adapter_ctx(ggml_init({2 * ggml_tensor_overhead(), nullptr, true}));
                auto * a = ggml_new_tensor_2d(adapter_ctx.get(), GGML_TYPE_F32, output->ne[0], 2);
                auto * b = ggml_new_tensor_2d(adapter_ctx.get(), GGML_TYPE_F32, 2, output->ne[1]);
                ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(adapter_ctx.get(), ggml_backend_cpu_buffer_type()));
                std::vector<float> a_data(ggml_nelements(a)), b_data(ggml_nelements(b));
                for (size_t i = 0; i < a_data.size(); ++i) { a_data[i] = 0.1f * std::sin(float(i)); }
                for (size_t i = 0; i < b_data.size(); ++i) { b_data[i] = 0.1f * std::cos(float(i)); }
                ggml_backend_tensor_set(a, a_data.data(), 0, ggml_nbytes(a));
                ggml_backend_tensor_set(b, b_data.data(), 0, ggml_nbytes(b));
                adapter.ab_map.emplace("output.weight", llama_adapter_lora_weight{a, b});
                llama_adapter_lora * adapters[] = {&adapter};
                float scale = 1.0f;
                require(llama_set_adapters_lora(ctx.get(), adapters, 1, &scale) != 0, "adding adapter with live KV rejected");
                llama_memory_clear(mem, false);
                require(llama_set_adapters_lora(ctx.get(), adapters, 1, &scale) == 0, "fixed adapter installed");
                const auto adapted = run(ctx.get(), input, 0, true);
                require(nmse(adapted, reference) > 1e-8, "adapter changes model output");
                require(llama_set_adapters_lora(ctx.get(), adapters, 1, &scale) == 0, "same adapter with live KV allowed");
                require(llama_set_adapters_lora(ctx.get(), nullptr, 0, nullptr) != 0, "removing live adapter rejected");
                std::vector<uint8_t> adapted_state(llama_state_get_size(ctx.get()));
                require(llama_state_get_data(ctx.get(), adapted_state.data(), adapted_state.size()) == adapted_state.size(), "save adapted state");
                const auto adapted_answer = run(ctx.get(), answer, input.size(), false);
                require(llama_state_set_data(ctx.get(), adapted_state.data(), adapted_state.size()) == adapted_state.size(), "restore matching adapter");
                equal(adapted_answer, run(ctx.get(), answer, input.size(), false));
                llama_memory_clear(mem, false);
                llama_adapter_lora reloaded(model);
                reloaded.ab_map = adapter.ab_map;
                reloaded.alpha = adapter.alpha;
                llama_adapter_lora * reloaded_adapters[] = {&reloaded};
                require(llama_set_adapters_lora(ctx.get(), reloaded_adapters, 1, &scale) == 0, "equivalent adapter object installed");
                require(llama_state_set_data(ctx.get(), adapted_state.data(), adapted_state.size()) == adapted_state.size(), "adapter signature independent of pointer");
                equal(adapted_answer, run(ctx.get(), answer, input.size(), false));
                scale = 0.5f;
                require(llama_set_adapters_lora(ctx.get(), adapters, 1, &scale) != 0, "live scale change rejected");
                llama_memory_clear(mem, false);
                require(llama_set_adapters_lora(ctx.get(), adapters, 1, &scale) == 0, "scale change after clear allowed");
                require(llama_state_set_data(ctx.get(), adapted_state.data(), adapted_state.size()) == 0, "adapter mismatch rejected");
                require(llama_set_adapters_lora(ctx.get(), nullptr, 0, nullptr) == 0, "adapter removed after clear");
                adapter.alora_invocation_tokens = {input[0]};
                require(llama_set_adapters_lora(ctx.get(), adapters, 1, &scale) != 0, "activated LoRA rejected");
            }
            equal(reference, run(ctx.get(), input, 0, true));
            {
                const int width = llama_model_n_embd(model);
                const int layers = llama_model_n_layer(model);
                std::vector<float> control(size_t(width) * (layers - 1));
                for (size_t i = 0; i < control.size(); ++i) { control[i] = 0.01f * std::sin(float(i)); }
                require(llama_set_adapter_cvec(ctx.get(), control.data(), control.size(), width, 1, layers - 1) != 0,
                        "live control vector change rejected");
                llama_memory_clear(mem, false);
                require(llama_set_adapter_cvec(ctx.get(), control.data(), control.size(), width, 1, layers - 1) == 0,
                        "fixed control vector installed");
                const auto controlled = run(ctx.get(), input, 0, true);
                require(nmse(controlled, reference) > 0, "control vector affects output");
                require(llama_set_adapter_cvec(ctx.get(), control.data(), control.size(), width, 1, layers - 1) == 0,
                        "same live control vector accepted");
                require(llama_set_adapter_cvec(ctx.get(), control.data(), width, width, 1, layers - 1) == 0,
                        "same live partial control vector update accepted");
                require(llama_set_adapter_cvec(ctx.get(), nullptr, 0, width, 1, layers - 1) != 0,
                        "live control vector removal rejected");
                std::vector<uint8_t> state(llama_state_get_size(ctx.get()));
                require(llama_state_get_data(ctx.get(), state.data(), state.size()) == state.size(), "save controlled state");
                llama_memory_clear(mem, false);
                require(llama_set_adapter_cvec(ctx.get(), nullptr, 0, width, 1, layers - 1) == 0, "remove cleared control vector");
                require(llama_state_set_data(ctx.get(), state.data(), state.size()) == 0, "control vector mismatch rejected");
                require(llama_set_adapter_cvec(ctx.get(), control.data(), control.size(), width, 1, layers - 1) == 0, "reload control vector");
                require(llama_state_set_data(ctx.get(), state.data(), state.size()) == state.size(), "restore controlled state");
                const auto * saved = llama_get_logits_seq(ctx.get(), 0);
                require(saved != nullptr, "saved boundary logits present");
                equal(std::vector<float>(controlled.end() - n_vocab, controlled.end()), std::vector<float>(saved, saved + n_vocab));
                llama_memory_clear(mem, false);
                require(llama_set_adapter_cvec(ctx.get(), control.data(), 1, width, 1, layers - 1) != 0, "short control vector rejected");
                require(llama_set_adapter_cvec(ctx.get(), nullptr, 0, width, 1, layers - 1) == 0, "disable control vector");
            }
            equal(reference, run(ctx.get(), input, 0, true));
            const auto pos = llama_memory_seq_pos_max(mem, 0);
            auto oversized = get_tokens(65, n_vocab, seed);
            require(llama_decode_prefix(ctx.get(), llama_batch_get_one(oversized.data(), oversized.size())) == -1, "no split prefix");
            require(llama_memory_seq_pos_max(mem, 0) == pos, "invalid prefix preserves memory");
            llama_token bad = n_vocab;
            require(llama_decode_prefix(ctx.get(), llama_batch_get_one(&bad, 1)) == -1, "invalid token rejected");
            auto batch = llama_batch_init(1, 0, 1);
            common_batch_add(batch, input[0], 0, {1}, true);
            require(llama_decode_prefix(ctx.get(), batch) == -1, "foreign sequence rejected");
            batch.seq_id[0][0] = 0;
            require(llama_decode(ctx.get(), batch) == -1, "rewriting prefix rejected");
            llama_batch_free(batch);
            require(!llama_memory_seq_rm(mem, 0, 1, -1), "partial removal rejected");
            require(!llama_memory_can_shift(mem), "shift capability disabled");
            llama_memory_seq_add(mem, 0, 0, -1, 1);
            llama_memory_seq_div(mem, 0, 0, -1, 2);
            llama_memory_seq_cp(mem, 0, 1, 0, -1);
            require(llama_memory_seq_pos_max(mem, 0) == pos && llama_memory_seq_pos_max(mem, 1) == -1, "unsafe memory operations preserve state");
            require(llama_state_get_size(ctx.get()) > 0, "state saving supported");
            require(llama_state_seq_get_size(ctx.get(), 0) > 0, "sequence saving supported");
            require(llama_state_set_data(ctx.get(), nullptr, 0) == 0, "state loading rejected before reading input");
            require(llama_state_seq_set_data(ctx.get(), nullptr, 0, 0) == 0, "sequence loading rejected");
            llama_set_causal_attn(ctx.get(), false);
            require(llama_get_attention_type(ctx.get()) == LLAMA_ATTENTION_TYPE_PREFIX_LM, "raw phase override rejected");
            equal(chunk, run(ctx.get(), answer, input.size(), false));
            const int next = input.size() + answer.size();
            require(llama_memory_seq_rm(mem, 0, 2, 2), "empty prefix interval is a no-op");
            require(llama_memory_seq_rm(mem, -1, 3, 1), "reversed interval is a no-op");
            require(llama_memory_seq_rm(mem, 0, next, -1), "removal after answer is a no-op");
            require(!llama_memory_seq_rm(mem, 0, input.size(), next - 1), "answer hole rejected");
            require(!llama_memory_seq_rm(mem, -1, 0, input.size()), "prefix-only removal rejected");
            llama_memory_seq_keep(mem, 0);
            llama_memory_seq_cp(mem, 0, 0, 1, 3);
            require(llama_memory_seq_pos_max(mem, 0) == next - 1, "safe memory operations preserve positions");
            for (const auto range : std::vector<std::pair<int, int>>{{0, -1}, {-1, next}, {0, next + 10}}) {
                const int trim = input.size() + 1;
                require(llama_memory_seq_rm(mem, range.first, trim, range.second), "answer suffix removed");
                require(llama_memory_seq_pos_max(mem, 0) == trim - 1, "suffix positions updated");
                equal(std::vector<float>(chunk.begin() + n_vocab, chunk.end()),
                      run(ctx.get(), std::vector<llama_token>(answer.begin() + 1, answer.end()), trim, false));
            }
            require(llama_memory_seq_rm(mem, -1, input.size(), -1), "all answer tokens removed");
            equal(chunk, run(ctx.get(), answer, input.size(), false));
            require(llama_memory_seq_rm(mem, 0, -1, next), "finite full removal");
            require(llama_memory_seq_pos_max(mem, 0) == -1, "full removal clears KV");
            require(llama_decode(ctx.get(), llama_batch_get_one(answer.data(), 1)) == -1, "full removal invalidates phase");
            require(llama_memory_seq_rm(mem, 0, 1, 2), "empty request removal is a no-op");
            equal(reference, run(ctx.get(), input, 0, true));
            llama_memory_clear(mem, true);
            require(llama_decode(ctx.get(), llama_batch_get_one(answer.data(), answer.size())) == -1, "clear invalidates prefix state");
            run(ctx.get(), {input[0]}, 0, true);
            run(ctx.get(), {answer[0]}, 1, false);
            auto full = get_tokens(64, n_vocab, seed);
            run(ctx.get(), full, 0, true);
            run(ctx.get(), full, 64, false);
            require(llama_decode(ctx.get(), llama_batch_get_one(answer.data(), 1)) == -1, "context overflow rejected");
            bool abort = true;
            llama_set_abort_callback(ctx.get(), [](void * data) { return *static_cast<bool *>(data); }, &abort);
            require(llama_decode_prefix(ctx.get(), llama_batch_get_one(input.data(), input.size())) == 2, "abort returned");
            require(llama_memory_seq_pos_max(mem, 0) == -1, "abort clears all KV");
            require(llama_decode(ctx.get(), llama_batch_get_one(answer.data(), 1)) == -1, "aborted prefix cannot continue");
            abort = false;
            equal(reference, run(ctx.get(), input, 0, true));
            abort = true;
            require(llama_decode(ctx.get(), llama_batch_get_one(answer.data(), answer.size())) == 2, "answer abort returned");
            require(llama_memory_seq_pos_max(mem, 0) == -1, "answer abort clears prefix and answer KV");
            abort = false;
            equal(reference, run(ctx.get(), input, 0, true));
            llama_set_abort_callback(ctx.get(), nullptr, nullptr);
            params.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
            llama_context_ptr bidir(llama_init_from_model(model, params));
            require(bool(bidir), "bidirectional context");
            equal(reference, run(bidir.get(), input, 0, false));
            require(llama_decode_prefix(bidir.get(), llama_batch_get_one(input.data(), input.size())) == -1, "prefix API requires prefix mode");
            params.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;
            llama_context_ptr causal(llama_init_from_model(model, params));
            require(bool(causal), "causal context");
            auto causal_ref = run(causal.get(), input, 0, false);
            llama_memory_clear(llama_get_memory(causal.get()), true);
            auto causal_changed = run(causal.get(), changed_input, 0, false);
            equal(std::vector<float>(causal_ref.begin(), causal_ref.begin() + n_vocab),
                  std::vector<float>(causal_changed.begin(), causal_changed.begin() + n_vocab));
            params.attention_type = LLAMA_ATTENTION_TYPE_PREFIX_LM;
            params.n_seq_max = 2;
            params.n_ctx = 256;
            for (bool unified : {false, true}) {
                params.kv_unified = unified;
                llama_context_ptr multi(llama_init_from_model(model, params));
                require(bool(multi), "parallel PrefixLM context");
                auto * mm = llama_get_memory(multi.get());
                require(llama_n_ctx_seq(multi.get()) == (unified ? 256u : 128u), "per-sequence capacity");
                const llama_seq_id other = unified ? 7 : 1;
                auto other_input = std::vector<llama_token>(changed_input.begin(), changed_input.begin() + 5);
                auto other_ref = run(ctx.get(), other_input, 0, true);
                auto other_answer = run(ctx.get(), answer, other_input.size(), false);
                // One logical batch combines a new prefix, its own answer, and another live answer.
                run(multi.get(), other_input, 0, true, other);
                auto phase_batch = llama_batch_init(input.size() + 2 * answer.size(), 0, 1);
                for (size_t i = 0; i < input.size(); ++i) { common_batch_add(phase_batch, input[i], i, {0}, true); }
                for (size_t i = 0; i < answer.size(); ++i) {
                    common_batch_add(phase_batch, answer[i], input.size() + i, {0}, true);
                    common_batch_add(phase_batch, answer[i], other_input.size() + i, {other}, true);
                }
                const llama_seq_id new_seq = 0;
                const llama_pos end = input.size();
                require(llama_decode_prefix_mixed(multi.get(), phase_batch, &new_seq, &end, 1) == 0, "mixed prefix and answer phases");
                for (size_t i = 0; i < size_t(phase_batch.n_tokens); ++i) {
                    const auto & want = i < input.size() ? reference : ((i - input.size()) % 2 ? other_answer : chunk);
                    const size_t row = i < input.size() ? i : (i - input.size()) / 2;
                    const float * got = llama_get_logits_ith(multi.get(), i);
                    equal(std::vector<float>(want.begin() + row * n_vocab, want.begin() + (row + 1) * n_vocab),
                          std::vector<float>(got, got + n_vocab));
                }
                const llama_pos absent_end = input.size() + answer.size() + 1;
                require(llama_decode_prefix_mixed(multi.get(), phase_batch, &new_seq, &absent_end, 1) == -1,
                        "incomplete declared prefix rejected before mutation");
                require(llama_memory_seq_pos_max(mm, other) == 8, "invalid mixed batch preserves unrelated answer");
                llama_batch_free(phase_batch);
                llama_memory_clear(mm, false);
                if (unified) {
                    auto shared = llama_batch_init(input.size() + answer.size(), 0, 2);
                    for (size_t i = 0; i < input.size(); ++i) { common_batch_add(shared, input[i], i, {0, other}, true); }
                    require(llama_decode_prefix(multi.get(), shared) == 0, "shared complete prefix");
                    for (size_t i = 0; i < input.size(); ++i) {
                        const auto * got = llama_get_logits_ith(multi.get(), i);
                        equal(std::vector<float>(reference.begin() + i*n_vocab, reference.begin() + (i+1)*n_vocab),
                              std::vector<float>(got, got + n_vocab));
                    }
                    common_batch_clear(shared);
                    for (size_t i = 0; i < answer.size(); ++i) { common_batch_add(shared, answer[i], input.size()+i, {other, 0}, true); }
                    require(llama_decode(multi.get(), shared) == 0, "shared causal answers with reordered owners");
                    for (size_t i = 0; i < answer.size(); ++i) {
                        const auto * got = llama_get_logits_ith(multi.get(), i);
                        equal(std::vector<float>(chunk.begin() + i*n_vocab, chunk.begin() + (i+1)*n_vocab),
                              std::vector<float>(got, got + n_vocab));
                    }
                    require(llama_get_logits_seq(multi.get(), 0) && llama_get_logits_seq(multi.get(), other), "logits saved for all owners");
                    std::vector<uint8_t> ownership(llama_state_get_size(multi.get()));
                    require(llama_state_get_data(multi.get(), ownership.data(), ownership.size()) == ownership.size(), "save shared ownership");
                    llama_memory_clear(mm, false);
                    require(llama_state_set_data(multi.get(), ownership.data(), ownership.size()) == ownership.size(), "restore shared ownership");

                    common_batch_clear(shared);
                    for (size_t i = 0; i + 1 < input.size(); ++i) { common_batch_add(shared, input[i], i, {0, other}, true); }
                    common_batch_add(shared, input.back(), input.size()-1, {0}, true);
                    common_batch_add(shared, input.back(), input.size()-1, {other}, true);
                    require(llama_decode_prefix(multi.get(), shared) == -1, "shared prefix with different future dependencies rejected");
                    require(llama_memory_seq_pos_max(mm, 0) == 11 && llama_memory_seq_pos_max(mm, other) == 11, "invalid shared prefix preserves owners");
                    common_batch_clear(shared);
                    common_batch_add(shared, answer[0], 12, {0, 0}, true);
                    require(llama_decode(multi.get(), shared) == -1, "duplicate owner rejected");
                    shared.seq_id[0][1] = other;
                    require(llama_decode(multi.get(), shared) == 0, "shared answer after state restore");
                    llama_memory_clear(mm, false);
                    run(multi.get(), input, 0, true);
                    auto different = input; different.back() = (different.back()+1) % n_vocab;
                    run(multi.get(), different, 0, true, other);
                    shared.pos[0] = input.size();
                    require(llama_decode(multi.get(), shared) == -1, "equal positions do not prove identical histories");
                    llama_memory_seq_cp(mm, 0, other, input.size(), input.size()+1);
                    require(llama_memory_seq_pos_max(mm, other) == 7, "empty answer range is a no-op");
                    llama_batch_free(shared);
                    llama_memory_clear(mm, false);
                }
                // Persistence and forks reuse the public state APIs and must preserve phase and branch isolation.
                equal(reference, run(multi.get(), input, 0, true));
                llama_memory_seq_cp(mm, 0, other, 1, -1);
                require(llama_memory_seq_pos_max(mm, other) == -1, "partial fork rejected");
                llama_memory_seq_cp(mm, 0, other, 0, input.size());
                require(llama_memory_seq_pos_max(mm, other) == 7, "complete finite-range fork");
                // Save immediately: separate-stream copy buffers have not yet been used by decode.
                std::vector<uint8_t> seq_state(llama_state_seq_get_size(multi.get(), other));
                require(llama_state_seq_get_data(multi.get(), seq_state.data(), seq_state.size(), other) == seq_state.size(), "save pending fork");
                equal(chunk, run(multi.get(), answer, input.size(), false, other));
                equal(chunk, run(multi.get(), answer, input.size(), false));
                llama_memory_seq_cp(mm, 0, other, 0, input.size() - 1);
                require(llama_memory_seq_pos_max(mm, other) == 11, "incomplete prefix copy preserves destination");
                llama_memory_seq_cp(mm, 0, other, 0, input.size() + 1);
                require(llama_memory_seq_pos_max(mm, other) == 8, "bounded copy includes complete prefix and answer head");
                require(!llama_get_logits_seq(multi.get(), other), "bounded copy invalidates unavailable boundary logits");
                equal(std::vector<float>(chunk.begin() + n_vocab, chunk.end()),
                      run(multi.get(), std::vector<llama_token>(answer.begin()+1, answer.end()), input.size()+1, false, other));
                if (unified) {
                    llama_memory_seq_cp(mm, 0, other, 0, input.size());
                    llama_memory_seq_cp(mm, 0, other, input.size(), input.size()+2);
                    require(llama_memory_seq_pos_max(mm, other) == 9, "answer-only copy with identical preceding KV");
                    equal(std::vector<float>(chunk.begin() + 2*n_vocab, chunk.end()),
                          run(multi.get(), std::vector<llama_token>(answer.begin()+2, answer.end()), input.size()+2, false, other));
                }
                llama_memory_seq_cp(mm, 0, other, 0, -1);
                require(llama_memory_seq_rm(mm, other, input.size(), -1), "fork after answers preserves prefix boundary");
                equal(changed, run(multi.get(), changed_answer, input.size(), false, other));
                require(llama_memory_seq_pos_max(mm, 0) == 11, "branch rollback preserves source answer");
                require(llama_state_seq_set_data(multi.get(), seq_state.data(), seq_state.size(), other) == seq_state.size(), "restore fork");
                require(llama_memory_seq_pos_max(mm, 0) == 11, "sequence restore preserves source");
                equal(chunk, run(multi.get(), answer, input.size(), false, other));
                require(llama_state_seq_set_data(multi.get(), seq_state.data(), seq_state.size(), 0) == seq_state.size(), "restore remaps sequence ID");
                equal(chunk, run(multi.get(), answer, input.size(), false));
                auto bad_state = seq_state;
                require(llama_state_seq_get_size(multi.get(), -1) == 0, "negative sequence state ID rejected");
                require(llama_state_seq_get_size_ext(multi.get(), other, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) == 0, "device snapshot flags rejected");
                // Prefix envelope: magic, source ID, version, signature, count, then ID/end/next.
                const size_t boundary_offset = 4 * sizeof(uint32_t) + sizeof(uint64_t) + sizeof(llama_seq_id);
                const llama_pos invalid_end = 0;
                std::memcpy(bad_state.data() + boundary_offset, &invalid_end, sizeof(invalid_end));
                require(llama_state_seq_set_data(multi.get(), bad_state.data(), bad_state.size(), other) == 0, "invalid prefix boundary rejected");
                require(llama_memory_seq_pos_max(mm, other) == 11, "metadata rejection preserves destination");
                bad_state = seq_state;
                // Corrupt the second KV position into a duplicate of the first (both endpoints stay intact).
                const size_t kv_offset = boundary_offset + 2 * sizeof(llama_pos);
                const size_t first_position = kv_offset + (unified ? 2 : 3) * sizeof(uint32_t);
                const llama_pos duplicate_pos = 0;
                std::memcpy(bad_state.data() + first_position + 3 * sizeof(uint32_t), &duplicate_pos, sizeof(duplicate_pos));
                require(llama_state_seq_set_data(multi.get(), bad_state.data(), bad_state.size(), other) == 0, "duplicate interior KV position rejected");
                require(llama_memory_seq_pos_max(mm, other) == -1 && llama_memory_seq_pos_max(mm, 0) == 11,
                        "corrupt KV isolates destination");
                require(llama_state_seq_set_data(multi.get(), seq_state.data(), seq_state.size(), other) == seq_state.size(), "restore after corrupt KV");
                equal(chunk, run(multi.get(), answer, input.size(), false, other));
                bad_state = seq_state;
                bad_state[0] ^= 1;
                require(llama_state_seq_set_data(multi.get(), bad_state.data(), bad_state.size(), other) == 0, "bad state magic rejected");
                require(llama_memory_seq_pos_max(mm, other) == 11, "header rejection preserves destination");
                require(llama_state_seq_set_data(multi.get(), seq_state.data(), seq_state.size() - 1, other) == 0, "truncated KV rejected");
                require(llama_memory_seq_pos_max(mm, other) == -1 && llama_memory_seq_pos_max(mm, 0) == 11,
                        "failed sequence restore invalidates destination only");
                require(llama_state_seq_set_data(multi.get(), seq_state.data(), seq_state.size(), other) == seq_state.size(), "restore after failure");
                std::vector<uint8_t> full_state(llama_state_get_size(multi.get()));
                require(llama_state_get_data(multi.get(), full_state.data(), full_state.size()) == full_state.size(), "save concurrent context");
                llama_memory_clear(mm, false);
                require(llama_state_set_data(multi.get(), full_state.data(), full_state.size()) == full_state.size(), "restore concurrent context");
                require(llama_memory_seq_pos_max(mm, 0) == 11 && llama_memory_seq_pos_max(mm, other) == 7, "restore independent phases");
                equal(chunk, run(multi.get(), answer, input.size(), false, other));
                require(llama_memory_seq_rm(mm, 0, input.size(), -1), "restored prefix boundary permits suffix trim");
                require(!llama_memory_seq_rm(mm, 0, input.size() - 1, -1), "restored prefix boundary rejects edit");
                equal(chunk, run(multi.get(), answer, input.size(), false));
                require(llama_state_set_data(multi.get(), full_state.data(), full_state.size() - 1) == 0, "truncated context rejected");
                require(llama_memory_seq_pos_max(mm, 0) == -1 && llama_memory_seq_pos_max(mm, other) == -1,
                        "failed context restore invalidates all sequences");
                llama_context_ptr restored(llama_init_from_model(model, params));
                require(llama_state_set_data(restored.get(), full_state.data(), full_state.size()) == full_state.size(), "restore into fresh context");
                const auto * restored_logits = llama_get_logits_seq(restored.get(), other);
                require(restored_logits != nullptr, "fresh-context boundary logits restored");
                equal(std::vector<float>(reference.end() - n_vocab, reference.end()),
                      std::vector<float>(restored_logits, restored_logits + n_vocab));
                equal(chunk, run(restored.get(), answer, input.size(), false, other));
                const std::string state_file = "test-prefix-state-" + std::string(llm_arch_name(arch)) + "-" + std::to_string(seed) + ".bin";
                run(multi.get(), input, 0, true);
                require(llama_state_save_file(multi.get(), state_file.c_str(), input.data(), input.size()), "save context file");
                llama_memory_clear(mm, false);
                std::vector<llama_token> saved_tokens(input.size());
                size_t saved_count = 0;
                require(llama_state_load_file(multi.get(), state_file.c_str(), saved_tokens.data(), saved_tokens.size(), &saved_count), "load context file");
                require(saved_tokens == input && saved_count == input.size(), "file tokens round trip");
                equal(chunk, run(multi.get(), answer, input.size(), false));
                run(multi.get(), input, 0, true);
                require(llama_state_seq_save_file(multi.get(), state_file.c_str(), 0, input.data(), input.size()) > 0, "save sequence file");
                require(llama_state_seq_load_file(multi.get(), state_file.c_str(), other, saved_tokens.data(), saved_tokens.size(), &saved_count) > 0, "load sequence file with remap");
                equal(chunk, run(multi.get(), answer, input.size(), false, other));
                std::remove(state_file.c_str());
                // A new prefix on one fork must not change the source's KV.
                equal(other_ref, run(multi.get(), other_input, 0, true, other));
                equal(chunk, run(multi.get(), answer, input.size(), false));
                auto causal_params = params;
                causal_params.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;
                llama_context_ptr causal(llama_init_from_model(model, causal_params));
                require(llama_state_set_data(causal.get(), full_state.data(), full_state.size()) == 0, "causal context rejects PrefixLM state");
                require(llama_state_seq_set_data(causal.get(), seq_state.data(), seq_state.size(), 0) == 0, "causal context rejects PrefixLM sequence state");
                run(causal.get(), input, 0, false);
                run(causal.get(), other_input, 0, false, other);
                std::vector<uint8_t> causal_state(llama_state_get_size(causal.get()));
                require(llama_state_get_data(causal.get(), causal_state.data(), causal_state.size()) == causal_state.size(), "causal state still saves");
                require(llama_state_set_data(multi.get(), causal_state.data(), causal_state.size()) == 0, "PrefixLM rejects causal context state");
                require(llama_state_set_data(causal.get(), causal_state.data(), causal_state.size()) == causal_state.size(), "causal state still restores");
                require(llama_memory_seq_pos_max(llama_get_memory(causal.get()), other) == llama_pos(other_input.size() - 1), "causal sparse ID state preserved");
                require(llama_memory_seq_rm(mm, other, 0, -1), "clear sequence before empty save");
                std::vector<uint8_t> empty_state(llama_state_seq_get_size(multi.get(), other));
                require(llama_state_seq_get_data(multi.get(), empty_state.data(), empty_state.size(), other) == empty_state.size(), "save empty sequence");
                require(llama_state_seq_set_data(multi.get(), empty_state.data(), empty_state.size(), 0) == empty_state.size(), "empty restore clears destination");
                require(llama_memory_seq_pos_max(mm, 0) == -1, "empty restore clears KV and phase");
                llama_memory_clear(mm, false);
                equal(reference, run(multi.get(), input, 0, true));
                equal(other_ref, run(multi.get(), other_input, 0, true, other));
                require(llama_memory_seq_pos_max(mm, 0) == 7, "new prefix preserves other sequence");
                auto mixed = llama_batch_init(2 * answer.size(), 0, 2);
                for (size_t i = 0; i < answer.size(); ++i) {
                    common_batch_add(mixed, answer[i], input.size() + i, {0}, true);
                    common_batch_add(mixed, answer[i], other_input.size() + i, {other}, true);
                }
                require(llama_decode(multi.get(), mixed) == 0, "interleaved answer batch");
                for (size_t i = 0; i < 2 * answer.size(); ++i) {
                    const auto & want = i % 2 ? other_answer : chunk;
                    const auto * got = llama_get_logits_ith(multi.get(), i);
                    equal(std::vector<float>(want.begin() + (i / 2) * n_vocab, want.begin() + (i / 2 + 1) * n_vocab),
                          std::vector<float>(got, got + n_vocab));
                }
                common_batch_clear(mixed);
                common_batch_add(mixed, answer[0], 12, {0}, true);
                common_batch_add(mixed, answer[0], 0, {other}, true);
                require(llama_decode(multi.get(), mixed) == -1, "mixed invalid admission is atomic");
                mixed.n_seq_id[0] = 2;
                mixed.seq_id[0][1] = other;
                require(llama_decode(multi.get(), mixed) == -1, "shared token with unequal histories rejected");
                llama_batch_free(mixed);
                require(!llama_memory_seq_rm(mm, -1, 6, -1), "wildcard invalid edit rejects atomically");
                require(llama_memory_seq_pos_max(mm, 0) == 11 && llama_memory_seq_pos_max(mm, other) == 8,
                        "wildcard rejection preserves both sequences");
                require(llama_memory_seq_rm(mm, other, other_input.size(), -1), "local answer rollback");
                equal(other_answer, run(multi.get(), answer, other_input.size(), false, other));
                equal(reference, run(multi.get(), input, 0, true));
                require(llama_memory_seq_pos_max(mm, other) == 8, "reset preserves other answer");
                bool cancelled = true;
                llama_set_abort_callback(multi.get(), [](void * data) { return *static_cast<bool *>(data); }, &cancelled);
                auto failed = llama_batch_init(input.size(), 0, 1);
                for (size_t i = 0; i < input.size(); ++i) { common_batch_add(failed, input[i], i, {other}, true); }
                require(llama_decode_prefix(multi.get(), failed) == 2, "local prefix abort");
                require(llama_memory_seq_pos_max(mm, other) == -1 && llama_memory_seq_pos_max(mm, 0) == 7,
                        "abort isolates unrelated sequence");
                cancelled = false;
                equal(chunk, run(multi.get(), answer, input.size(), false));
                // Complete prefixes may also share a batch, with separate sequence ownership.
                common_batch_clear(failed);
                for (size_t i = 0; i < 4; ++i) {
                    common_batch_add(failed, input[i], i, {0}, true);
                    common_batch_add(failed, input[i], i, {other}, true);
                }
                require(llama_decode_prefix(multi.get(), failed) == 0, "multiple complete prefixes");
                auto short_ref = run(ctx.get(), std::vector<llama_token>(input.begin(), input.begin() + 4), 0, true);
                for (int i = 0; i < 8; ++i) {
                    const auto * got = llama_get_logits_ith(multi.get(), i);
                    equal(std::vector<float>(short_ref.begin() + (i / 2) * n_vocab, short_ref.begin() + (i / 2 + 1) * n_vocab),
                          std::vector<float>(got, got + n_vocab));
                }
                // Missing positions are inferred independently for each sequence.
                auto * saved_pos = failed.pos;
                failed.pos = nullptr;
                require(llama_decode(multi.get(), failed) == 0, "implicit interleaved positions");
                failed.pos = saved_pos;
                cancelled = true;
                common_batch_clear(failed);
                common_batch_add(failed, answer[0], 8, {other}, true);
                require(llama_decode(multi.get(), failed) == 2, "local answer abort");
                cancelled = false;
                require(llama_memory_seq_pos_max(mm, 0) == 7 && llama_memory_seq_pos_max(mm, other) == -1,
                        "answer abort isolates unrelated sequence");
                require(llama_decode(multi.get(), failed) == -1, "aborted sequence needs a new prefix");
                llama_batch_free(failed);
                run(multi.get(), other_input, 0, true, other);
                llama_memory_seq_keep(mm, other);
                require(llama_memory_seq_pos_max(mm, 0) == (unified ? -1 : 7), "keep follows cache stream ownership");
                require(llama_decode(multi.get(), llama_batch_get_one(answer.data(), 1)) == (unified ? -1 : 0),
                        "keep phase follows retained KV");
                equal(other_answer, run(multi.get(), answer, other_input.size(), false, other));
                require(llama_memory_seq_rm(mm, -1, 0, -1), "wildcard full removal");
                require(llama_memory_seq_pos_max(mm, other) == -1, "wildcard removal clears sequences");
                auto limited_params = params;
                limited_params.n_ctx = 17;
                llama_context_ptr limited(llama_init_from_model(model, limited_params));
                require(bool(limited), "small parallel capacity");
                run(limited.get(), input, 0, true);
                run(limited.get(), other_input, 0, true, other);
                if (unified) { run(limited.get(), answer, input.size(), false); }
                auto overflow = llama_batch_init(1, 0, 1);
                common_batch_add(overflow, answer[0], unified ? 5 : 8, {unified ? other : 0}, true);
                require(llama_decode(limited.get(), overflow) == -1, "requested capacity enforced before padded allocation");
                llama_batch_free(overflow);
                require(llama_memory_seq_pos_max(llama_get_memory(limited.get()), 0) == (unified ? 11 : 7),
                        "capacity rejection preserves sequence");
            }
        }
    }
    printf("%d PrefixLM engine checks passed (%s)\n", checks, llm_arch_name(arch));
    return 0;
}

int main(int argc, char ** argv) {
    // init the logger at max verbosity. filter with a custom callback respecting the user-configure verbosity
    common_log_set_verbosity_thold(LOG_LEVEL_DEBUG);
    common_init();

    std::random_device rd;

    llm_arch arch = LLM_ARCH_UNKNOWN;
    size_t seed = rd();
    std::string out;

    int verbosity = LOG_LEVEL_ERROR;
    bool prefix_lm = false;
    std::string prefix_reference;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--prefix-reference") == 0 && i + 1 < argc) { prefix_reference = argv[++i]; continue; }
        if (strcmp(argv[i], "--prefix-lm") == 0) { prefix_lm = true; }
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv);
            return 0;
        }
        if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--arch") == 0) {
            if (i + 1 < argc) {
                const std::string arch_name = argv[++i];
                arch = llm_arch_from_string(arch_name);
                if (arch == LLM_ARCH_UNKNOWN) {
                    LOG_ERR("%s: unkown LLM architecture: %s\n", __func__, arch_name.c_str());
                    return 1;
                }
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc) {
                seed = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-v") == 0) {
            if (i + 1 < argc) {
                verbosity = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--out") == 0) {
            if (i + 1 < argc) {
                out = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        }
    }
    printf("%s: using seed %zu\n", __func__, seed);

    try {
        if (!prefix_reference.empty()) { return test_prefix_reference(prefix_reference); }
        if (prefix_lm) { return test_prefix_lm(arch, seed); }
        if (!out.empty()) {
            return save_models(arch, seed, verbosity, out);
        }
        return test_backends(arch, seed, verbosity);
    } catch (const std::exception & err) {
        fprintf(stderr, "encountered runtime error: %s\n", err.what());
        return -1;
    }
}
