# awesome-long-term-memory
A collection of long-term memory papers for large language models (LLMs), organized by their primary focus.

### Methods (organized by evaluation task)

#### Long-Context
1. [Transformer-XL: Attentive Language Models beyond a Fixed-Length Context](https://aclanthology.org/P19-1285/) (Dai et al., ACL 2019)  
2. [Compressive Transformers for Long-Range Sequence Modelling](https://openreview.net/forum?id=SylVNerFvr) (Rae et al., ICLR 2020)  
3. [Recurrent Memory Transformer](https://papers.nips.cc/paper_files/paper/2022/hash/e4b3215d2741ea9a748ee2b70b0c197e-Abstract-Conference.html) (Bulatov et al., NeurIPS 2022)  
4. [Memformer: A Memory-Augmented Transformer for Sequence Modeling](https://aclanthology.org/2022.findings-aacl.29/) (Wu et al., AACL 2022)  
5. [Augmenting Language Models with Long-Term Memory](https://arxiv.org/abs/2306.07174) (Wang et al., NeurIPS 2023)  
6. [HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models](https://arxiv.org/abs/2405.14831) (Jiménez Gutiérrez et al., NeurIPS 2024)  
7. [MATTER: Memory-Augmented Transformer Using Heterogeneous Knowledge Sources](https://aclanthology.org/2024.findings-acl.81/) (Lee et al., ACL 2024 Findings)  
8. [MemoryLLM: Towards Self-Updatable Large Language Models](https://arxiv.org/abs/2402.04624) (Wang et al., 2024)  
9. [MemLong: Memory-Augmented Retrieval for Long Text Modeling](https://arxiv.org/abs/2408.16967) (Liu et al., 2024)  
10. [Memory Layers at Scale](https://arxiv.org/abs/2412.09764) (Berges et al., 2024)  
11. [Memory Augmented Large Language Models are Computationally Universal](https://arxiv.org/abs/2301.04589) (Schuurmans, 2023)  
12. [M+: Extending MemoryLLM with Scalable Long-Term Memory](https://openreview.net/forum?id=OcqbkROe8J) (Wang et al., ICML 2025)  
13. [From RAG to Memory: Non-Parametric Continual Learning for LLMs](https://openreview.net/forum?id=LWH8yn4HS2) (Jiménez Gutiérrez et al., ICML 2025)  
14. [Can Memory-Augmented Language Models Generalize on Reasoning-in-a-Haystack Tasks?](https://arxiv.org/abs/2503.07903) (Das et al., 2025) – **MemReasoner architecture**  

#### Chat Assistant
1. [MemoryBank: Enhancing Large Language Models with Long-Term Memory](https://doi.org/10.1609/aaai.v38i17.29946) (Zhong et al., AAAI 2024)  
2. [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) (Packer et al., 2023)  
3. [In Prospect and Retrospect: Reflective Memory Management for Long-Term Personalized Dialogue Agents](https://aclanthology.org/2025.acl-long.413/) (Tan et al., ACL 2025)  
4. [Prime: LLM Personalization with Cognitive Memory and Thought Processes](https://arxiv.org/abs/2507.04607) (Zhang et al., 2025)  
5. [MemAgent: Reshaping Long-Context LLM with Multi-Conversation RL-Based Memory Agent](https://arxiv.org/abs/2507.02259) (Yu et al., 2025)  

#### Reasoning
1. [Sleep-Time Compute: Beyond Inference Scaling at Test-Time](https://arxiv.org/abs/2504.13171) (Lin et al., 2025)  
2. [ReasonIR: Training Retrievers for Reasoning Tasks](https://openreview.net/forum?id=kkBCNLMbGj) (Shao et al., LM-Conf 2025)  
3. [Emergent Hierarchical Reasoning in LLMs through Reinforcement Learning](https://arxiv.org/abs/2509.03646) (H. Wang et al., 2025)  
4. [Contextual Experience Replay for Self-Improvement of Language Agents](https://aclanthology.org/2025.acl-long.694/) (Y. Liu et al., ACL 2025)  

#### Agent & Multi-Modal Systems
1. [ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory](https://arxiv.org/abs/2509.25140) (Ouyang et al., 2025)  
2. [Agent Workflow Memory](https://openreview.net/forum?id=NTAhi2JEEE) (Z. Wang et al., ICML 2025)  
3. [Agent KB: Leveraging Cross-Domain Experience for Agentic Problem Solving](https://arxiv.org/abs/2507.06229) (X. Tang et al., 2025)  
4. [Mem¹: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents](https://arxiv.org/abs/2506.15841) (Zhou et al., 2025)  
5. [Synapse: Trajectory-as-Exemplar Prompting with Memory for Computer Control](https://openreview.net/forum?id=Pc8AU1aF5e) (Zheng et al., ICLR 2024)  
6. [MemTool: Optimizing Short-Term Memory Management for Dynamic Tool Calling in LLM Agents](https://arxiv.org/abs/2507.21428) (Lumer et al., 2025)  

### Benchmarks
* LoCoMo — [arXiv](https://arxiv.org/abs/2402.17753), [project](https://snap-research.github.io/locomo/), [ACL Anthology](https://aclanthology.org/2024.acl-long.747/)
* LongMemEval — [arXiv](https://arxiv.org/abs/2410.10813), [GitHub](https://github.com/xiaowu0162/LongMemEval), [website](https://xiaowu0162.github.io/long-mem-eval/)
* BFCL V4 — [leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html), [memory blog](https://gorilla.cs.berkeley.edu/blogs/16_bfcl_v4_memory.html), [Gorilla repo](https://github.com/ShishirPatil/gorilla)
* PerLTQA — [arXiv](https://arxiv.org/abs/2402.16288), [ACL Anthology](https://aclanthology.org/2024.sighan-1.18/), [GitHub](https://github.com/Elvin-Yiming-Du/PerLTQA)
* LongBench — [arXiv](https://arxiv.org/abs/2308.14508), [ACL Anthology](https://aclanthology.org/2024.acl-long.172/), [GitHub](https://github.com/THUDM/LongBench)
* RULER — [arXiv](https://arxiv.org/abs/2404.06654), [GitHub](https://github.com/NVIDIA/RULER), [OpenReview](https://openreview.net/forum?id=kIoBbc76Sy)
* ∞Bench / InfiniteBench — [arXiv](https://arxiv.org/abs/2402.13718), [ACL Anthology](https://aclanthology.org/2024.acl-long.814/), [GitHub](https://github.com/OpenBMB/InfiniteBench)
* L-Eval — [arXiv](https://arxiv.org/abs/2307.11088), [OpenReview](https://openreview.net/forum?id=eUAr4HwU0X), [GitHub](https://github.com/OpenLMLab/LEval)
* Needle-in-a-Haystack — [GitHub](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)
* HELMET — [arXiv](https://arxiv.org/abs/2410.02694), [OpenReview](https://openreview.net/forum?id=293V3bJbmE), [GitHub](https://github.com/princeton-nlp/HELMET), [project](https://princeton-nlp.github.io/HELMET/)

### Resources
* Zep — [GitHub](https://github.com/getzep/zep), [Python SDK](https://github.com/getzep/zep-python), [Go SDK](https://github.com/getzep/zep-go)
* Letta (formerly MemGPT) — [GitHub](https://github.com/letta-ai/letta), [website](https://www.letta.com/)
* Mem0 — [GitHub](https://github.com/mem0ai/mem0), [website](https://mem0.ai/)
* LangChain / LangGraph memory — [LangMem GitHub](https://github.com/langchain-ai/langmem), [LangGraph memory docs](https://langchain-ai.github.io/langgraph/concepts/memory/)
* LlamaIndex memory — [docs](https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/memory/)
* MemGPT (project page) — [Berkeley project](https://sky.cs.berkeley.edu/project/memgpt/)
