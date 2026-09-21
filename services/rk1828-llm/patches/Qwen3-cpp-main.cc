// Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>

#include <string>
#include <vector>
#include <iostream>

#include "rknn_qwen3_llm.h"
#include "image_utils.h"
#include "time_utils.h"

int64_t first_token;
bool first_decode = true;

// ── RK1828 Qwen3 LLM server mode (opt-in via trailing "-" argv) ──────────
// Modelled 1:1 on the already-shipped gemma4 / Qwen3-TTS server modes.
//
//   argv (server) : <model_dir> [--model-name <stem>] [--core-mask <hex>]
//                   [--max-context <n>] [--kv-checkpoint-interval <n>]
//                   [--kv-checkpoint-count <n>] [--device-id <id>] -
//                   <stem> defaults to Qwen3-4B; Qwen3.5 exports load the same
//                   way (identical RKNN3 LLM API), e.g. --model-name Qwen3.5-4B
//   argv (one-shot, unchanged):
//                   <model_path> <weight_path> <tokenizer_path>
//                   <embedding_path> <core_mask> <prompt>
//
//   stderr : "READY 1" once Init completes (handshake + protocol version).
//            All RKNN verbose diagnostics also land here.
//   stdin  : one request line per turn, TAB delimited (no JSON dependency).
//            Two accepted forms; escaping is the same for both -- the client
//            escapes "\\" -> backslash, "\n" -> newline, "\t" -> tab, so one
//            request is always exactly one line.
//
//              legacy: <max_new_tokens>\t<prompt>
//              V2:     V2\t<max_new>\t<keep>\t<tools_json>\t<n>\t(<role>\t<text>)*n
//
//            V2 carries the OpenAI-style tools array. The runtime renders it
//            through the model's OWN Jinja chat template (minja, from the GGUF),
//            so the canonical Qwen3 tool preamble lands in a real system block
//            instead of being hand-written into the user turn. Empty tools_json
//            leaves the current tool set alone; it is only re-registered when it
//            changes, since each call re-renders the template.
//
//            n is accepted for forward compatibility but the runtime rejects
//            more than one input per run ("Only support one LLM input!"), so
//            callers must flatten history into a single turn. Likewise `keep`
//            (keep_history) works but makes the session stateful, which a single
//            shared EP cannot do safely under concurrency -- it stays 0 here.
//   stdout : per generated token  [uint32 LE len][utf8 token bytes]
//            per request end      [uint32 LE 0xFFFFFFFE]  (EOS sentinel)
//            failed request       [uint32 LE 0xFFFFFFFD]  then the EOS sentinel
//   EOF on stdin -> clean exit.
//
// stdout discipline: the real stdout fd is dup'd to g_frame_fd BEFORE model
// init and the C stdout FILE* is re-pointed at stderr, so no printf() anywhere
// in this file or in librknn3 can ever inject stray bytes into the frame
// channel (a single stray byte desyncs the reader into a GB-sized read).
#define QWEN3_LLM_PROTOCOL_VERSION 1
static const uint32_t LLM_END_OF_STREAM = 0xFFFFFFFEu;
// Sent before the EOS frame when the request failed (rknn3_session_run != 0),
// so the client can tell a failure from an empty answer. Measured on the rk3588
// devkit (2026-09-21): a prompt longer than the model's KV cache fails the run,
// and without this marker the client got HTTP 200 with empty content.
static const uint32_t LLM_REQUEST_FAILED = 0xFFFFFFFDu;
static std::string g_last_tools;
static bool        g_kv_reuse = true;  // RK1828_KV_REUSE=0 disables prefix reuse
static bool g_server_mode = false;
static int  g_frame_fd    = -1;

static void frame_write_all(const void* buf, size_t len)
{
  const char* p = (const char*)buf;
  while (len > 0) {
    ssize_t n = write(g_frame_fd, p, len);
    if (n <= 0) {
      if (n < 0 && errno == EINTR) continue;
      fprintf(stderr, "[server] frame write failed (n=%zd errno=%d)\n", n, errno);
      return;
    }
    p += n;
    len -= (size_t)n;
  }
}

static void emit_token_frame(const std::string& piece)
{
  if (piece.empty()) return;  // zero-length frames carry no text; reader tolerates
  uint32_t len = (uint32_t)piece.size();
  frame_write_all(&len, sizeof(len));
  frame_write_all(piece.data(), piece.size());
}

static void emit_failed_frame()
{
  uint32_t marker = LLM_REQUEST_FAILED;
  frame_write_all(&marker, sizeof(marker));
}

static void emit_eos_frame()
{
  uint32_t marker = LLM_END_OF_STREAM;
  frame_write_all(&marker, sizeof(marker));
}

static std::string unescape_request(const std::string& in)
{
  std::string out;
  out.reserve(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    if (in[i] == '\\' && i + 1 < in.size()) {
      char c = in[++i];
      if (c == 'n')       out.push_back('\n');
      else if (c == 't')  out.push_back('\t');
      else if (c == 'r')  out.push_back('\r');
      else if (c == '\\') out.push_back('\\');
      else                { out.push_back('\\'); out.push_back(c); }
    } else {
      out.push_back(in[i]);
    }
  }
  return out;
}

static std::string join_path(const std::string& dir, const char* name)
{
  if (dir.empty()) return std::string(name);
  if (dir.back() == '/') return dir + name;
  return dir + "/" + name;
}

struct embedding_info
{
  int      fd;
  float16* embedding_data;
  int      embedding_dim;
  int      vocab_size;
};

const rknn3_sampling_params SAMPLE_PARAMS = {
    .top_k = 1,
    .top_p = 0.9,
    .temperature = 1.0f,
    .repeat_penalty = 1.2f,
    .frequency_penalty = 0.0f,
    .presence_penalty = 0.0f
};

const char* system_prompt  = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n";
const char* prompt_prefix  = "<|im_start|>user\n";
const char* prompt_postfix = "<|im_end|>\n<|im_start|>assistant\n";

/*-------------------------------------------
                Callback Function
-------------------------------------------*/
int result_callback(void *userdata, RKLLMResult *result, LLMCallState state)
{
    Tokenizer *tokenizer = (Tokenizer *)userdata;

    if (state == RKLLM_RUN_ERROR)
    {
        printf("\n\nError occurred during inference\n");
        return 0;
    }
    else if (state == RKLLM_RUN_FINISH)
    {
        printf("\n\n--------------------Finished-------------------- \n");
        return 0;
    }
    else if (state == RKLLM_RUN_WAITING)
    {
        printf("\n\nWaiting for UTF-8 encoded character\n");
        return 0;
    }
    else if (state == RKLLM_RUN_MAX_NEW_TOKEN_REACHED)
    {
        printf("\n\n--------------Max new token reached------------- \n");
        return 0;
    }
    else if (state == RKLLM_RUN_STOP)
    {
        printf("\n\n-----------------------Stop--------------------- \n");
        return 0;
    }
    else if (state == RKLLM_RUN_NORMAL)
    {
        // Get token text
        std::string piece;
        if (result->num_tokens == 1) {
          piece = tokenizer->TokenToPiece(result->token_ids[0]);
        } else {
          piece = tokenizer->Decode(result->token_ids, result->num_tokens);
        }

        if (g_server_mode) {
            emit_token_frame(piece);
        } else {
            printf("%s", piece.c_str());
        }

        if (first_decode) {
            first_token = getCurrentTimeUs();
            first_decode = false;
        }
        fflush(stdout);
    }
    return 0;
}

int tokenizer_callback(void *userdata, const char *text, int32_t text_len, int32_t *tokens, int32_t n_tokens_max)
{
    int n_tokens = 0;
    Tokenizer *tokenizer = (Tokenizer *)userdata;
    n_tokens = tokenizer->Tokenize(text, text_len, tokens, n_tokens_max);

    if (n_tokens <= 0)
    {
        printf("tokenizer failed for %s\n", text);
        return n_tokens;
    }

    return n_tokens;
}

int embed_callback(void* userdata, int32_t* tokens, uint64_t num_tokens, void* embed, uint64_t len)
{
    struct embedding_info* embed_info = (struct embedding_info*)userdata;

    if (len != num_tokens * embed_info->embedding_dim * sizeof(float16)) {
        printf("invalid embed buffer\n");
        return -1;
    }

    for (int n = 0; n < num_tokens; n++) {
        // Same guard as the model zoo's Qwen3.5 demo: a token id outside the
        // embedding table would read past the mmap.
        if (tokens[n] < 0 || tokens[n] >= embed_info->vocab_size) {
            printf("embed_callback: token id %d out of range [0, %d)\n", tokens[n], embed_info->vocab_size);
            return -1;
        }
        memcpy((unsigned char*)embed + n * embed_info->embedding_dim * sizeof(float16), embed_info->embedding_data + tokens[n] * embed_info->embedding_dim,
                embed_info->embedding_dim * sizeof(float16));
    }

    return 0;
}

void printf_perf(rknn_perf_metrics_t *p)
{

    printf("\n--------------------------------------------------------------------------------------\n");
    printf(" %-12s  %-15s  %-8s  %-23s  %-23s\n",
           "Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second");
    printf("--------------------------------------------------------------------------------------\n");


    float ttft_us = (float)(first_token - p->llm_start_time);
    int prefill_n_tokens = p->n_prefill_tokens;
    float prefill_ms = ttft_us / 1000.0;
    float prefill_tpt = prefill_n_tokens == 0 ? 0.0f : prefill_ms / prefill_n_tokens;
    float prefill_tps = prefill_n_tokens == 0 ? 0.0f : 1e3f / prefill_ms * prefill_n_tokens;
    printf(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f\n",
           "Prefill", prefill_ms, prefill_n_tokens, prefill_tpt, prefill_tps);

    float decode_time_us = (float)(p->llm_end_time - first_token);
    float decode_ms = decode_time_us / 1000.0;
    int decode_n_tokens = p->n_decode_tokens;
    float decode_tpt = decode_n_tokens == 0 ? 0.0f : decode_ms / decode_n_tokens;
    float decode_tps = decode_n_tokens == 0 ? 0.0f : 1e3f / decode_ms * decode_n_tokens;
    printf(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f\n",
           "Generate", decode_ms, decode_n_tokens, decode_tpt, decode_tps);

    printf("--------------------------------------------------------------------------------------\n");
}

// Per-request inference with a caller-supplied max_new_tokens (the shared
// inference_qwen3_llm() in rknn_qwen3_llm.cc hardcodes MAX_NEW_TOKENS).
// One conversation turn to submit: role ("" = user) plus its text.
struct ServerTurn {
    std::string role;
    std::string text;
};

static int server_infer(rknn_qwen3_llm_context* llm_ctx,
                        const std::vector<ServerTurn>& turns,
                        int32_t max_new_tokens, rknn_perf_metrics_t* perf,
                        int keep_history = 0)
{
    if (!llm_ctx || !llm_ctx->rknn_sess) {
        fprintf(stderr, "[server] session is NULL\n");
        return -1;
    }
    if (turns.empty()) return -1;
    // The runtime rejects more than one input per run with
    // "RKLLMSession: Only support one LLM input!" (measured 2026-07-31), so a
    // whole conversation cannot be submitted at once even though the API takes
    // an array. Callers flatten history into a single turn; keep the last one
    // and say so rather than letting the opaque RKNNAPI error surface.
    std::vector<ServerTurn> use;
    if (turns.size() > 1) {
        fprintf(stderr, "[server] WARNING: %zu inputs requested but the runtime "
                        "accepts only 1; using the last turn\n", turns.size());
        use.push_back(turns.back());
    } else {
        use = turns;
    }
    const std::vector<ServerTurn>& turns_in = use;

    std::vector<rknn3_llm_input> inputs(turns_in.size());
    rknn3_llm_infer_param llm_infer_param;
    memset(&llm_infer_param, 0, sizeof(llm_infer_param));

    for (size_t i = 0; i < turns_in.size(); ++i) {
        rknn3_llm_tensor tensor;
        memset(&tensor, 0, sizeof(tensor));
        tensor.name            = "input_embeds";
        tensor.prompt          = turns_in[i].text.c_str();
        tensor.embed           = NULL;
        tensor.tokens          = NULL;
        tensor.n_tokens        = 0;
        tensor.enable_thinking = false;

        memset(&inputs[i], 0, sizeof(inputs[i]));
        inputs[i].input_type = RKNN3_LLM_INPUT_PROMPT;
        inputs[i].llm_input  = tensor;
        // role picks how the runtime's Jinja template renders this turn:
        // "tool" -> <tool_response>, empty/NULL -> a normal user turn.
        inputs[i].role       = turns_in[i].role.empty() ? NULL : turns_in[i].role.c_str();
    }

    llm_infer_param.keep_history   = keep_history;
    llm_infer_param.max_new_tokens = max_new_tokens;

    first_decode = true;
    first_token  = 0;

    perf->llm_start_time = getCurrentTimeUs();
    int ret = rknn3_session_run(llm_ctx->rknn_sess, inputs.data(),
                                (uint32_t)inputs.size(), &llm_infer_param);
    perf->llm_end_time = getCurrentTimeUs();
    if (ret < 0) {
        fprintf(stderr, "[server] rknn3_session_run fail ret=%d\n", ret);
        return ret;
    }

    RKLLMRunState state = {0};
    if (rknn3_session_query_state(llm_ctx->rknn_sess, &state) >= 0) {
        perf->n_decode_tokens  = state.n_decode_tokens;
        perf->n_prefill_tokens = state.n_prefill_tokens;
    }
    return ret;
}

// Unconditional KV clear after EVERY request (including error paths): skipped
// clears accumulate dirty KV, overflow max_context and SIGABRT at runtime.
static void server_reset_kvcache(rknn_qwen3_llm_context* llm_ctx,
                                 rknn3_kvcache_clear_policy policy = RKNN3_KVCACHE_CLEAR_ALL)
{
    if (!llm_ctx || !llm_ctx->rknn_sess) return;
    int ret = rknn3_session_clear_kvcache(llm_ctx->rknn_sess, policy);
    if (ret != RKNN3_SUCCESS) {
        fprintf(stderr, "[server] clear_kvcache failed ret=%d\n", ret);
    }
}

static int run_server(const std::string& model_dir, const std::string& model_name,
                      uint32_t core_mask, int32_t max_context_len,
                      int64_t ckpt_interval, int64_t ckpt_count)
{
    // Re-route stdout FIRST, before any model init chatter: dup the real stdout
    // for raw frames, point the C stdout FILE* at stderr.
    g_frame_fd = dup(STDOUT_FILENO);
    if (g_frame_fd < 0) {
        fprintf(stderr, "[server] dup(stdout) failed\n");
        return -1;
    }
    if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
        fprintf(stderr, "[server] dup2 stderr->stdout failed\n");
        return -1;
    }

    // The four files of one export share a stem (the model zoo writes
    // <name>.rknn / .weight / .tokenizer.gguf / .embed.bin). Qwen3 and Qwen3.5
    // use the same RKNN3 LLM API, so the model is chosen here, not by a
    // separate binary.
    std::string model_path     = join_path(model_dir, (model_name + ".rknn").c_str());
    std::string weight_path    = join_path(model_dir, (model_name + ".weight").c_str());
    std::string tokenizer_path = join_path(model_dir, (model_name + ".tokenizer.gguf").c_str());
    std::string embedding_path = join_path(model_dir, (model_name + ".embed.bin").c_str());

    int ret = 0;
    rknn_qwen3_llm_context rknn_app_ctx;
    Tokenizer*             tokenizer = NULL;
    VocabInfo              vocab_info;
    struct embedding_info  embedding_info;
    struct stat            emb_st;
    rknn3_llm_param        params;
    RKLLMCallback          callback;

    memset(&rknn_app_ctx, 0, sizeof(rknn_app_ctx));
    memset(&vocab_info, 0, sizeof(vocab_info));
    memset(&embedding_info, 0, sizeof(embedding_info));
    memset(&emb_st, 0, sizeof(emb_st));
    memset(&params, 0, sizeof(params));
    memset(&callback, 0, sizeof(callback));
    embedding_info.fd = -1;
    embedding_info.embedding_data = NULL;

    fprintf(stderr, "[server] model_dir=%s model_name=%s core_mask=0x%x max_context_len=%d\n",
            model_dir.c_str(), model_name.c_str(), core_mask, max_context_len);

    tokenizer = new Tokenizer(TOKENIZER_BACKEND_LLAMA, tokenizer_path.c_str());
    if (!tokenizer) {
        fprintf(stderr, "[server] load tokenizer failed: %s\n", tokenizer_path.c_str());
        ret = -1;
        goto srv_out;
    }
    tokenizer->GetVocabInfo(&vocab_info);
    fprintf(stderr, "[server] vocab_size=%d n_eos=%d n_bos=%d\n",
            vocab_info.vocab_size, vocab_info.n_special_eos_id, vocab_info.n_special_bos_id);

    embedding_info.fd = open(embedding_path.c_str(), O_RDONLY);
    if (embedding_info.fd == -1) {
        fprintf(stderr, "[server] open embedding failed: %s\n", embedding_path.c_str());
        ret = -1;
        goto srv_out;
    }
    if (fstat(embedding_info.fd, &emb_st) == -1) {
        fprintf(stderr, "[server] fstat embedding failed\n");
        ret = -1;
        goto srv_out;
    }
    embedding_info.embedding_data =
        (float16*)mmap(NULL, emb_st.st_size, PROT_READ, MAP_PRIVATE, embedding_info.fd, 0);
    if (embedding_info.embedding_data == MAP_FAILED) {
        fprintf(stderr, "[server] mmap embedding failed\n");
        embedding_info.embedding_data = NULL;
        ret = -1;
        goto srv_out;
    }
    embedding_info.vocab_size    = vocab_info.vocab_size;
    embedding_info.embedding_dim = (emb_st.st_size / vocab_info.vocab_size) / sizeof(float16);

    params.logits_name                 = (char*)"logits";
    params.max_context_len             = max_context_len;
    params.sampling_param              = SAMPLE_PARAMS;
    params.vocab_info.vocab_size       = vocab_info.vocab_size;
    params.vocab_info.n_special_eos_id = vocab_info.n_special_eos_id;
    params.vocab_info.n_special_bos_id = vocab_info.n_special_bos_id;
    memcpy(params.vocab_info.special_eos_id, vocab_info.special_eos_id, sizeof(vocab_info.special_eos_id));
    memcpy(params.vocab_info.special_bos_id, vocab_info.special_bos_id, sizeof(vocab_info.special_bos_id));

    callback.result_callback    = result_callback;
    callback.result_userdata    = tokenizer;
    callback.tokenizer_callback = tokenizer_callback;
    callback.tokenizer_userdata = tokenizer;
    callback.embed_callback     = embed_callback;
    callback.embed_userdata     = &embedding_info;

    fprintf(stderr, "[server] --> init qwen3 llm model\n");
    ret = init_qwen3_llm(&rknn_app_ctx, model_path.c_str(), weight_path.c_str(),
                         &params, 1, callback, core_mask);
    if (ret != 0) {
        fprintf(stderr, "[server] init_qwen3_llm fail ret=%d\n", ret);
        goto srv_out;
    }

    {
        const char* v = getenv("RK1828_KV_REUSE");
        if (v && v[0] == '0') g_kv_reuse = false;
        fprintf(stderr, "[server] kv prefix reuse: %s\n", g_kv_reuse ? "on" : "off");
    }

    // Linear-attention models (Qwen3.5) do not reuse a shared prompt prefix on
    // their own: without checkpoints every request re-prefills the whole
    // system prompt (rk3588 devkit, 2026-09-21: 1687 ms TTFT on every request
    // vs 132 ms for Qwen3-4B). SAVE_CHECKPOINT snapshots the state every
    // <interval> tokens and later requests resume from the last snapshot inside
    // the shared prefix (RKNN3 V1.1.0 dev guide 4.2.6). The runtime ignores it
    // for full-attention models, and aligns/clamps the values itself.
    if (ckpt_interval > 0 && ckpt_count > 0) {
        rknn3_kvcache_policy_param kp;
        memset(&kp, 0, sizeof(kp));
        kp.save_checkpoint.checkpoint_start_pos      = 0;
        kp.save_checkpoint.checkpoint_interval       = ckpt_interval;
        kp.save_checkpoint.max_checkpoint_count      = ckpt_count;
        kp.save_checkpoint.checkpoint_tail_overwrite = false;
        int kret = rknn3_session_set_kvcache_policy(rknn_app_ctx.rknn_sess,
                                                    RKNN3_KVCACHE_POLICY_SAVE_CHECKPOINT, &kp);
        fprintf(stderr, "[server] kv checkpoint interval=%lld count=%lld ret=%d\n",
                (long long)ckpt_interval, (long long)ckpt_count, kret);
    }

    g_server_mode = true;
    fprintf(stderr, "[server] Init complete\n");
    fprintf(stderr, "READY %d\n", QWEN3_LLM_PROTOCOL_VERSION);
    fflush(stderr);

    {
        std::string line;
        int utt = 0;
        while (std::getline(std::cin, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
            if (line.empty()) {
                emit_eos_frame();
                continue;
            }

            int32_t                 req_max_new = 256;
            int                     req_keep    = 0;
            std::string             req_tools;
            std::vector<ServerTurn> req_turns;
            bool                    req_bad = false;
            // V2 request line, additive -- anything not starting with "V2\t"
            // takes the original single-prompt path untouched:
            //   V2 \t max_new \t keep \t esc(tools_json) \t n \t (role \t esc(text)) * n
            // tools_json empty = leave the current tool set alone.
            if (line.rfind("V2\t", 0) == 0) {
                std::vector<std::string> f;
                size_t pos = 3;
                while (true) {
                    size_t t = line.find('\t', pos);
                    if (t == std::string::npos) { f.push_back(line.substr(pos)); break; }
                    f.push_back(line.substr(pos, t - pos));
                    pos = t + 1;
                }
                if (f.size() >= 4) {
                    req_max_new = (int32_t)strtol(f[0].c_str(), NULL, 10);
                    if (req_max_new <= 0) req_max_new = 256;
                    req_keep  = (int)strtol(f[1].c_str(), NULL, 10);
                    req_tools = unescape_request(f[2]);
                    int n = (int)strtol(f[3].c_str(), NULL, 10);
                    if (n > 0 && (int)f.size() >= 4 + 2 * n) {
                        for (int i = 0; i < n; ++i) {
                            ServerTurn tn;
                            tn.role = f[4 + 2 * i];
                            tn.text = unescape_request(f[5 + 2 * i]);
                            req_turns.push_back(tn);
                        }
                    } else {
                        req_bad = true;
                    }
                } else {
                    req_bad = true;
                }
                if (req_bad) fprintf(stderr, "[server] malformed V2 line\n");
            } else {
                size_t tab = line.find('\t');
                ServerTurn tn;
                if (tab == std::string::npos) {
                    tn.text = unescape_request(line);
                } else {
                    req_max_new = (int32_t)strtol(line.substr(0, tab).c_str(), NULL, 10);
                    if (req_max_new <= 0) req_max_new = 256;
                    tn.text = unescape_request(line.substr(tab + 1));
                }
                req_turns.push_back(tn);
            }

            // Re-register tools only when the set actually changes: the runtime
            // re-renders its Jinja template on every call, and the advertised
            // set is stable for a whole session.
            // LIMITATION (RKNN3 V1.1.0, measured on the rk3588 devkit
            // 2026-09-21): once a session has registered tools it keeps them.
            // A later tool-less request is still rendered with the tool
            // preamble (prefill 189-252 tokens vs 28-29 on a fresh worker) and
            // the model may "call" tools the client never offered, which the
            // shim does not parse for that request. Registering "[]" or ""
            // returns 0 and changes nothing, and there is no unregister API.
            // So one server must not mix tool and tool-less clients; the
            // agent's warmup only sends tools its turns send (PR #113).
            if (!req_tools.empty() && req_tools != g_last_tools) {
                int tret = rknn3_session_set_function_tools(rknn_app_ctx.rknn_sess,
                                                            req_tools.c_str());
                fprintf(stderr, "[server] set_function_tools ret=%d bytes=%zu\n",
                        tret, req_tools.size());
                if (tret == 0) g_last_tools = req_tools;
            }

            std::string prompt = req_turns.empty() ? std::string() : req_turns.back().text;
            if (prompt.empty()) {
                fprintf(stderr, "[server] req#%d empty prompt\n", utt);
                server_reset_kvcache(&rknn_app_ctx);
                emit_eos_frame();
                utt++;
                continue;
            }

            fprintf(stderr, "[server] req#%d max_new=%d turns=%zu keep=%d prompt_bytes=%zu\n",
                    utt, req_max_new, req_turns.size(), req_keep, prompt.size());

            rknn_perf_metrics_t perf;
            memset(&perf, 0, sizeof(perf));
            // keep: 0 = stateless (production), 1 = keep_history on.
            // 2 (KEEP_SYSTEM_PROMPT) is deliberately not offered -- see below.
            int rc = server_infer(&rknn_app_ctx, req_turns, req_max_new, &perf,
                                  req_keep == 1 ? 1 : 0);
            if (rc != 0) {
                fprintf(stderr, "[server] req#%d FAILED rc=%d prefill_tokens=%d\n",
                        utt, rc, perf.n_prefill_tokens);
                emit_failed_frame();
            } else {
                float ttft_ms = first_token ? (float)(first_token - perf.llm_start_time) / 1000.0f : 0.0f;
                float dec_ms  = first_token ? (float)(perf.llm_end_time - first_token) / 1000.0f : 0.0f;
                fprintf(stderr, "[server] req#%d done prefill_tokens=%d ttft_ms=%.2f "
                                "decode_tokens=%d decode_ms=%.2f decode_tps=%.2f\n",
                        utt, perf.n_prefill_tokens, ttft_ms, perf.n_decode_tokens, dec_ms,
                        dec_ms > 0 ? 1e3f / dec_ms * perf.n_decode_tokens : 0.0f);
            }

            // Do NOT clear the KV cache between stateless requests.
            //
            // With keep_history=0 the runtime keeps an automatic PREFIX cache:
            // the next request reuses whatever leading tokens it shares with
            // this one. Every request repeats the same system block and tool
            // schema, so that prefix is most of the prompt. Measured on device
            // 2026-07-31, tool-calling requests:
            //
            //   clearing every time : 358-378 ms TTFT, every request
            //   not clearing        : 378 ms first, then 123 ms  (-255 ms)
            //
            // Verified safe over 12 consecutive DIFFERING requests: arguments
            // always tracked the current request (a prefix cache is not
            // conversation history, so nothing bleeds across), prefill stayed
            // flat at 208-209 tokens (no accumulation, so no drift toward
            // max_context) and no request came back empty.
            //
            // The comment that used to sit here called the unconditional clear
            // load-bearing against dirty-KV overflow. That was observed with
            // keep_history=1, where turns genuinely accumulate; it does not
            // apply to this path.
            //
            // RKNN3_KVCACHE_KEEP_SYSTEM_PROMPT is NOT used: on this runtime it
            // silently corrupts the session once function tools are registered
            // (the generation first comes back empty, then the model starts
            // reciting the tool preamble back as its answer). It is also
            // unnecessary now.
            //
            // RK1828_KV_REUSE=0 restores the old unconditional clear, as an
            // escape hatch if reuse is ever suspected in a misbehaviour.
            if (req_keep == 0 && !g_kv_reuse) {
                server_reset_kvcache(&rknn_app_ctx, RKNN3_KVCACHE_CLEAR_ALL);
            }
            emit_eos_frame();
            fflush(stderr);
            utt++;
        }
    }

    ret = 0;

srv_out:
    release_qwen3_llm(&rknn_app_ctx);
    if (embedding_info.embedding_data) {
        munmap((void*)embedding_info.embedding_data, emb_st.st_size);
        embedding_info.embedding_data = NULL;
    }
    if (embedding_info.fd != -1) {
        close(embedding_info.fd);
        embedding_info.fd = -1;
    }
    if (tokenizer) { delete tokenizer; tokenizer = NULL; }
    return ret;
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    // ── Server mode dispatch (opt-in): trailing "-" sentinel ──
    if (argc >= 3 && std::string(argv[argc - 1]) == "-")
    {
        std::string model_dir;
        std::string model_name      = "Qwen3-4B";
        int64_t     ckpt_interval   = 0;   // 0 = no KV checkpoints
        int64_t     ckpt_count      = 0;
        uint32_t    core_mask       = 0xff;
        int32_t     max_context_len = MAX_CONTEXT_LEN;

        if (const char* e = getenv("QWEN3_CORE_MASK"))   core_mask       = (uint32_t)strtoul(e, NULL, 16);
        if (const char* e = getenv("QWEN3_MAX_CONTEXT")) max_context_len = (int32_t)strtol(e, NULL, 10);

        for (int i = 1; i < argc - 1; ++i) {
            std::string a = argv[i];
            if (a == "--core-mask" && i + 1 < argc - 1) {
                core_mask = (uint32_t)strtoul(argv[++i], NULL, 16);
            } else if (a == "--max-context" && i + 1 < argc - 1) {
                max_context_len = (int32_t)strtol(argv[++i], NULL, 10);
            } else if (a == "--model-name" && i + 1 < argc - 1) {
                model_name = argv[++i];
            } else if (a == "--kv-checkpoint-interval" && i + 1 < argc - 1) {
                ckpt_interval = (int64_t)strtoll(argv[++i], NULL, 10);
            } else if (a == "--kv-checkpoint-count" && i + 1 < argc - 1) {
                ckpt_count = (int64_t)strtoll(argv[++i], NULL, 10);
            } else if (a == "--device-id" && i + 1 < argc - 1) {
                ++i;  // device selection is via PCIe enumeration in the runtime
            } else if (model_dir.empty()) {
                model_dir = a;
            }
        }
        if (model_dir.empty()) {
            fprintf(stderr, "server mode usage: %s <model_dir> [--model-name <stem>] "
                            "[--core-mask <hex>] [--max-context <n>] [--kv-checkpoint-interval <n>] "
                            "[--kv-checkpoint-count <n>] [--device-id <id>] -\n", argv[0]);
            return -1;
        }
        return run_server(model_dir, model_name, core_mask, max_context_len,
                          ckpt_interval, ckpt_count);
    }

    if (argc != 7)
    {
        printf("%s <model_path> <weight_path> <tokenizer_path> <embedding_path> <core_mask> <prompt>\n", argv[0]);
        printf("%s <model_dir> [--model-name <stem>] [--core-mask <hex>] [--max-context <n>] [--device-id <id>] -   (server mode)\n", argv[0]);
        return -1;
    }

    const char *model_path     = argv[1];
    const char *weight_path    = argv[2];
    const char *tokenizer_path = argv[3];
    const char *embedding_path = argv[4];
    uint32_t    core_mask      = strtoul(argv[5], nullptr, 16);
    const char *prompt         = argv[6];

    int ret;
    rknn_perf_metrics_t perf;

    // RKNN Context
    rknn_qwen3_llm_context rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_qwen3_llm_context));

    // Tokenizer
    Tokenizer* tokenizer;
    VocabInfo vocab_info;

    // Embedding
    struct embedding_info embedding_info;
    struct stat           emb_st;
    memset(&embedding_info, 0x00, sizeof(embedding_info));

    // LLM Param
    int n_params = 1;
    rknn3_llm_param params;
    memset(&params, 0, sizeof(params));

    // LLM Input Tensor
    int n_inputs = 1;
    rknn3_llm_tensor tensor;
    memset(&tensor, 0, sizeof(rknn3_llm_tensor));

    // Callback
    RKLLMCallback callback;
    memset(&callback, 0, sizeof(callback));

    // Load Tokenizer
    tokenizer = new Tokenizer(TOKENIZER_BACKEND_LLAMA, tokenizer_path);
    if (!tokenizer)
    {
        printf("load tokenizer failed! tokenizer_path=%s\n", tokenizer_path);
        goto out;
    }

    tokenizer->GetVocabInfo(&(vocab_info));
    printf("vocab_info: vocab_size=%d, special_bos_id=[", vocab_info.vocab_size);
    for (int i = 0; i < vocab_info.n_special_bos_id; ++i)
    {
        printf("%d%s", vocab_info.special_bos_id[i], (i + 1 < vocab_info.n_special_bos_id) ? ", " : "");
    }
    printf("], special_eos_id=[");
    for (int i = 0; i < vocab_info.n_special_eos_id; ++i)
    {
        printf("%d%s", vocab_info.special_eos_id[i], (i + 1 < vocab_info.n_special_eos_id) ? ", " : "");
    }
    printf("]\n");

    // Read Embedding
    embedding_info.fd = open(embedding_path, O_RDONLY);
    if (embedding_info.fd == -1) {
        printf("Failed to open embedding file: %s\n", embedding_path);
        goto out;
    }

    if (fstat(embedding_info.fd, &emb_st) == -1) {
        printf("Failed to get embedding file size\n");
        goto out;
    }

    embedding_info.embedding_data = (float16*)mmap(NULL, emb_st.st_size, PROT_READ, MAP_PRIVATE, embedding_info.fd, 0);
    if (embedding_info.embedding_data == MAP_FAILED) {
        printf("Failed to mmap embedding file\n");
        goto out;
    }

    embedding_info.vocab_size    = vocab_info.vocab_size;
    embedding_info.embedding_dim = (emb_st.st_size / vocab_info.vocab_size) / sizeof(float16);

    // Set LLM parameters
    params.logits_name               = "logits";
    params.max_context_len           = MAX_CONTEXT_LEN;
    params.sampling_param            = SAMPLE_PARAMS;
    params.vocab_info.vocab_size     = vocab_info.vocab_size;
    params.vocab_info.n_special_eos_id = vocab_info.n_special_eos_id;
    params.vocab_info.n_special_bos_id = vocab_info.n_special_bos_id;
    memcpy(params.vocab_info.special_eos_id, vocab_info.special_eos_id, sizeof(vocab_info.special_eos_id));
    memcpy(params.vocab_info.special_bos_id, vocab_info.special_bos_id, sizeof(vocab_info.special_bos_id));

    // LLM Callback
    callback.result_callback    = result_callback;
    callback.result_userdata    = tokenizer;
    callback.tokenizer_callback = tokenizer_callback;
    callback.tokenizer_userdata = tokenizer;
    callback.embed_callback     = embed_callback;
    callback.embed_userdata     = &embedding_info;

    printf("--> init qwen3 llm model\n");
    ret = init_qwen3_llm(&rknn_app_ctx, model_path, weight_path, &params, n_params, callback, core_mask);
    if (ret != 0)
    {
        printf("init_qwen3_llm fail! ret=%d model_path=%s weight_path=%s\n", ret, model_path, weight_path);
        goto out;
    }

    // LLM Input
    tensor.name     = "input_embeds";
    tensor.prompt   = prompt;
    tensor.embed    = NULL;
    tensor.tokens   = NULL;
    tensor.n_tokens = 0;
    tensor.enable_thinking = false;

    printf("--> inference qwen3 llm model\n");
    ret = inference_qwen3_llm(&rknn_app_ctx, tensor, n_inputs, &perf);
    if (ret != 0)
    {
        printf("inference qwen3 llm fail! ret=%d\n", ret);
        goto out;
    }

    printf_perf(&perf);

out:
    ret = release_qwen3_llm(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release qwen3 llm fail! ret=%d\n", ret);
    }

    if (embedding_info.fd != -1) {
        if (embedding_info.embedding_data != MAP_FAILED && embedding_info.embedding_data != NULL) {
            munmap((void*)embedding_info.embedding_data, emb_st.st_size);
            embedding_info.embedding_data = NULL;
        }
        close(embedding_info.fd);
        embedding_info.fd = -1;
    }

    if (tokenizer != NULL)
    {
        delete tokenizer;
        tokenizer = NULL;
    }

    return ret;
}
