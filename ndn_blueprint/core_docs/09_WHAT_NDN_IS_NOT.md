# What NDN Is Not

## NDN is not a monolithic LLM

NDN nodes are small (~70M parameter) encoder-decoder models. They do not reason, generate novel text, or understand instructions. They compress text into latent representations and reconstruct text from those representations. The agent's LLM (GPT, Claude, Qwen, etc.) does all reasoning. NDN is infrastructure, not intelligence.

## NDN is not AGI or a step toward AGI

NDN is a memory compression architecture. It stores and retrieves information for AI agents. It does not learn new capabilities at runtime. It does not generalize to tasks it was not trained on. It does not exhibit emergent reasoning. Calling it a step toward AGI would be dishonest.

## NDN is not a replacement for the orchestrator

An AI agent needs an orchestrator (the LLM that reads context, decides actions, and generates responses). NDN does not orchestrate. It feeds compressed memory into the orchestrator's context window. Without an orchestrator, NDN is just a database of tensors.

## NDN is not guaranteed to beat markdown on every memory task

On the OpenClaw AOJ journal A/B test, v4 reaches near-parity with markdown:
- Markdown: 100% fact recovery
- NDN AOJ v4 (single-session compression): 99% fact recovery, 1.78x compression
- NDN AOJ v2 (prior champion): 73% fact recovery, 1.69x compression — the 73% ceiling was a data problem, not architecture

On leaf-specific pipelines with entity side-channel at scale:
- TDR flagship leaf: 93–94% fact recovery at 89–105x compression (100 reports, 1.15M tokens)
- RWJ blooming: 84% fact recovery at 50x compression (257 files, 952K tokens). With v4 engine: dev 95% facts at 63x compression.

At the single-session level, v4 nearly closes the gap (99% vs 100%). At scale, where markdown is impossible (1M+ tokens), NDN's leaf pipelines provide competitive fact recovery with practical compression. NDN is not yet lossless, and the 1% gap at single-session and the larger gaps at scale mean markdown remains superior where context window is not a constraint. Data leakage caveat: v4's test journals were in training (same 5% overlap as v2).

## NDN is not permission to invent arbitrary taxonomy branches

Adding a new domain or subdomain to the NDN requires empirical evidence:
1. Existing proxy nodes must demonstrably fail on the target text type
2. Dedicated training must measurably improve results
3. The improvement must survive held-out evaluation

"This text type feels different" is not sufficient justification. "I think it would be cool to have a node for X" is not sufficient justification. Evidence of proxy failure and dedicated improvement is the minimum bar.

## NDN is not proof that training metrics mean real-world usefulness

The project has a documented case where internal metrics improved while real-world performance regressed:

- AOJ v3 internal: lower val_loss, earlier exact match, higher shuffled_gap than v2
- AOJ v3 real A/B: 66% fact recovery (down from v2's 73%)

Training metrics are necessary for monitoring convergence. They are not sufficient for claiming practical value. When they conflict with real-world A/B results, the A/B results take precedence for practical claims.

_**v4 postscript:** AOJ v4 (99% fact recovery) showed the v2/v3 gap was primarily a training data problem — real GitHub journal data closed most of the gap that loss weighting could not. However, v4's TDR scale result is slightly worse than v2 (17/20 vs 18/20 hits, 91% vs 93% facts), indicating that retrieval rather than compression is now the primary bottleneck. The lesson from v3 still holds: internal metrics alone are not sufficient evidence._

## NDN is not a finished product

NDN is a research project with working code and validated results. It is not production-ready software. It has not been deployed in a live agent system. The integration with OpenClaw was a test harness, not a production deployment. Using NDN in production today would require significant engineering beyond what currently exists.

## NDN is not a universal compression scheme

NDN compresses specific types of text using models trained on those types. It does not compress images, audio, video, structured databases, or binary data. It is not a general-purpose compression algorithm. It is a domain-specialized text memory system.

## NDN is not an alternative to vector retrieval

Vector retrieval (embed → store → search by similarity → return chunks) and NDN (classify → compress → store → retrieve → reconstruct) are complementary approaches, not competitors. Vector retrieval finds relevant information. NDN compresses it. A mature agent memory system might use both: vector retrieval for search, NDN for storage efficiency.

## NDN is not "solved"

The project has validated the core mechanism, tested it across multiple domains, produced a public benchmark result, and demonstrated real-world integration. But the primary open problem — rare entity preservation — is not solved. The routing layer is basic. The fusion layer is simple concatenation. There is no production deployment. Significant work remains before NDN can be claimed as a mature technology.
