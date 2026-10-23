# CS781 Phase 1: LegalGPT model families

The local archive contains three core families: A (12-layer dense causal Transformer, 110.1M parameters), B (6-layer encoder plus 6-layer cross-attending decoder, 124.3M), and C (10-layer prompt encoder plus 4-layer GRU generator, 117.8M). A LegalSpan extension adds legal-span routing and copying; an improved B continues the B architecture, rather than becoming a new family. B1 and C1 reruns use a handwritten BPE implementation.

All named complete archives passed ZIP integrity and manifest checks in the local audit. Historical Codabench screenshot scores were A0 0.8304, B0 0.7972, C 0.8237, LegalSpan 0.8341, improved B 0.8244. These are screenshot-attributed historical values, not freshly retrieved private leaderboard measurements. The first A0/B0/C/improved-B versions used a library BPE learner and failed the later own-BPE rule; LegalSpan, B1, and C1 used handwritten BPE. An instructor must decide novelty and full rule compliance.

Original model code, tokenizer assets, prepared legal corpora, prediction ZIPs, screenshots, and checkpoints remain in local archives and are excluded from this public snapshot pending course and data rights review.
