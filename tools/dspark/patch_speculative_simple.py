#!/usr/bin/env python3
"""Instrument llama.cpp examples/speculative-simple/speculative-simple.cpp (in /data/inkling-mini/llama-unsloth only):
  * DSPARK_STEPLOG=<file>: one line per verify step: STEP new= n_draft= n_acc= t_draft_us= t_target_us= t_proc_us=
  * rejection-sampling verification (same call as llama-server) when --spec-draft-sampling probabilistic and temp > 0
Idempotent: refuses to patch twice. Usage: /usr/bin/python3 -I patch_speculative_simple.py"""
import sys
F = "/data/inkling-mini/llama-unsloth/examples/speculative-simple/speculative-simple.cpp"
s = open(F).read()
if "DSPARK_STEPLOG" in s:
    print("already patched"); sys.exit(0)

def rep(old, new):
    global s
    assert s.count(old) == 1, ("pattern not unique/found", old[:60], s.count(old))
    s = s.replace(old, new)

rep("    llama_tokens draft;\n\n    common_prompt_checkpoint ckpt;\n",
    """    llama_tokens draft;

    // --- DSPARK instrumentation (agent R) ---
    std::vector<std::vector<llama_token_data>> draft_q;
    bool draft_is_replay = false;
    FILE * steplog = nullptr;
    if (const char * p = getenv("DSPARK_STEPLOG")) { steplog = fopen(p, "w"); }
    int64_t t_draft_us = 0;
    const bool use_rej_cfg = params.speculative.draft.probabilistic && params.sampling.temp > 0.0f;
    if (use_ckpt_tgt) { LOG_ERR("DSPARK instrumentation does not handle checkpoint replay\\n"); return 2; }

    common_prompt_checkpoint ckpt;
""")
rep("""            // generate a new draft
            common_speculative_get_draft_params(spec, seq_id) = {
                /* .drafting   = */ true,
                /* .n_max      = */ n_draft_max,
                /* .pos0       = */ n_past,
                /* .id_last    = */ id_last,
                /* .prompt     = */ &prompt_tgt,
                /* .result     = */ &draft, // output
            };
            common_speculative_draft(spec);
""", """            // generate a new draft
            draft_is_replay = false;
            draft_q.clear();
            common_speculative_get_draft_params(spec, seq_id) = {
                /* .drafting   = */ true,
                /* .n_max      = */ n_draft_max,
                /* .pos0       = */ n_past,
                /* .id_last    = */ id_last,
                /* .prompt     = */ &prompt_tgt,
                /* .result     = */ &draft, // output
                /* .result_q   = */ use_rej_cfg ? &draft_q : nullptr,
                /* .temp       = */ params.sampling.temp,
                /* .seed       = */ params.sampling.seed,
            };
            const int64_t t_d0 = ggml_time_us();
            common_speculative_draft(spec);
            t_draft_us = ggml_time_us() - t_d0;
""")
rep("""            // we have a previous (partial) draft to reuse from checkpoint restoration
""", """            // we have a previous (partial) draft to reuse from checkpoint restoration
            draft_is_replay = true;
""")
rep("""        // always have a token to evaluate from before - id_last
        batch_tgt.clear();""", """        // always have a token to evaluate from before - id_last
        const int64_t t_v0 = ggml_time_us();
        batch_tgt.clear();""")
rep("""            llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_tgt.get());
        }

        // feed the batch to the speculative implementation(s) - this drives the draft model, MTP, Eagle3, etc.
        if (!common_speculative_process(spec, batch_tgt)) {""", """            llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_tgt.get());
        }
        const int64_t t_v1 = ggml_time_us();

        // feed the batch to the speculative implementation(s) - this drives the draft model, MTP, Eagle3, etc.
        if (!common_speculative_process(spec, batch_tgt)) {""")
rep("""        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);
""", """        const int64_t t_v2 = ggml_time_us();
        std::vector<llama_token> ids;
        if (use_rej_cfg && !draft_q.empty() && draft_q.size() == draft.size()) {
            std::vector<int> idxs(draft.size() + 1);
            for (size_t i = 0; i < idxs.size(); ++i) { idxs[i] = (int) i; }
            ids = common_sampler_sample_and_accept_n_rejection(smpl.get(), ctx_tgt, idxs, draft, draft_q);
        } else {
            ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);
        }
        if (steplog) {
            fprintf(steplog, "STEP new=%d n_draft=%zu n_acc=%zu t_draft_us=%lld t_target_us=%lld t_proc_us=%lld rej=%d\\n",
                    draft_is_replay ? 0 : 1, n_draft, ids.size() - 1, (long long) t_draft_us, (long long) (t_v1 - t_v0),
                    (long long) (t_v2 - t_v1), (use_rej_cfg && !draft_q.empty()) ? 1 : 0);
            fflush(steplog);
        }
""")
rep("""    common_speculative_free(spec);

    llama_backend_free();""", """    if (steplog) { fclose(steplog); }
    common_speculative_free(spec);

    llama_backend_free();""")
open(F, "w").write(s)
print("patched")
