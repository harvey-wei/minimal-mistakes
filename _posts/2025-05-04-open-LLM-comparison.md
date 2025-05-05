# Comparison of Open-Source LLMs: LLaMA 3, Qwen, Gemma, DeepSeek

This report provides a detailed academic comparison of major open-source large language models (LLMs), focusing on architecture, training methods, and datasets. The models compared are:

- **LLaMA 3** by Meta
- **Qwen** by Alibaba
- **Gemma** by Google DeepMind
- **DeepSeek** by DeepSeek AI

---

## 1. Model Architecture Comparison

| Model       | Parameters (Open)          | Transformer Features                                   | Context Length | Tokenizer         | Notes |
|-------------|-----------------------------|--------------------------------------------------------|----------------|--------------------|-------|
| **LLaMA 3** | 1B to 405B (dense)          | Decoder-only, GQA, RMSNorm, SwiGLU, RoPE               | Up to 128K     | SentencePiece (128K) | Vision & edge variants |
| **Qwen 3**  | 0.5B to 235B (MoE)          | Decoder-only, GQA, MoE (128 experts), RMSNorm, RoPE    | Up to 128K     | Byte-level BPE (152K) | Multilingual |
| **Gemma**   | 1B to 27B (dense)           | Decoder-only, Multi-Query Attn (small), RMSNorm, GeGLU | Up to 128K     | SentencePiece (256K) | Vision-language in Gemma 3 |
| **DeepSeek**| 671B total, 37B active (MoE)| MoE, Multi-Head Latent Attention, RoPE                 | 128K           | SentencePiece       | High-performance RL-trained |

---

## 2. Training Methods

| Model       | Pre-training Tokens | Alignment & Fine-tuning              | Notable Techniques |
|-------------|---------------------|--------------------------------------|---------------------|
| **LLaMA 3** | >15T                | SFT + DPO/RLHF (LLaMA 3.1/3.2)       | Extended context, vision support |
| **Qwen 3**  | ~3T per model       | SFT + RLHF + Domain-specific tuning  | MoE, multilingual, tool use |
| **Gemma**   | 3T (2B), 6T (7B)    | SFT + Human-in-loop (no confirmed RLHF) | Long context, vision fine-tuning |
| **DeepSeek**| 14.8T               | RL-first, then SFT + RLHF            | Multi-stage RL, MTP, MoE distillation |

---

## 3. Training Data Sources

| Model       | Main Data Sources                     | Multilingual | Code | Math/Science | Filtering |
|-------------|----------------------------------------|--------------|------|---------------|-----------|
| **LLaMA 3** | Web, books, code, Wikipedia, papers    | ~8%          | ✓    | ✓             | Extensive |
| **Qwen 3**  | Web (Chinese/English), Wikipedia, code | ✓            | ✓    | ✓             | Aggressive |
| **Gemma**   | Web, books, code, technical content    | Moderate     | ✓    | ✓             | Strong filters |
| **DeepSeek**| Web, GitHub, arXiv, QA forums, code    | ✓            | ✓    | ✓✓            | Highly curated |

---

## 4. Summary

- **LLaMA 3**: Balanced general-purpose model, long context, excellent performance, vision support in 3.2.
- **Qwen 3**: Strong multilingual and reasoning, MoE support, tool-use ready.
- **Gemma**: Lightweight, great for research/teaching, includes vision-capable and long-context variants.
- **DeepSeek**: RL-driven reasoning, excellent performance, distilled for broader use.

---

## References

- Meta AI. LLaMA 3 Technical Report. 2024.
- Alibaba Cloud. Qwen Technical Documentation.
- Google DeepMind. Gemma Model Card. 2024–2025.
- DeepSeek AI. DeepSeek-R1 Report. 2025.
