# awesome-long-term-memory
A collection of long-term memory papers for large language models (LLMs), organized by their primary focus.

Author: [Di Wu](https://xiaowu0162.github.io/) and [GPT-5-Thinking](https://openai.com/index/introducing-gpt-5/). 

This readme only reflects what we found in our literture research, which is by no means complete. Feel free to submit a pr if you are interested!

### Methods (organized by evaluation task)

#### Long-Context
1. [Transformer-XL: Attentive Language Models beyond a Fixed-Length Context](https://aclanthology.org/P19-1285/) (Dai et al., ACL 2019)  
2. [Compressive Transformers for Long-Range Sequence Modelling](https://openreview.net/forum?id=SylVNerFvr) (Rae et al., ICLR 2020)  
3. [Memformer: A Memory-Augmented Transformer for Sequence Modeling](https://aclanthology.org/2022.findings-aacl.29/) (Wu et al., AACL 2022)  
4. [Recurrent Memory Transformer](https://papers.nips.cc/paper_files/paper/2022/hash/e4b3215d2741ea9a748ee2b70b0c197e-Abstract-Conference.html) (Bulatov et al., NeurIPS 2022)  
5. [Memory Augmented Large Language Models are Computationally Universal](https://arxiv.org/abs/2301.04589) (Schuurmans, arXiv 2023)  
6. [Augmenting Language Models with Long-Term Memory](https://arxiv.org/abs/2306.07174) (Wang et al., NeurIPS 2023)  
7. [MEMORYLLM: Towards Self-Updatable Large Language Models](https://arxiv.org/abs/2402.04624) (Wang et al., ICML 2024)  
8. [MATTER: Memory-Augmented Transformer Using Heterogeneous Knowledge Sources](https://aclanthology.org/2024.findings-acl.953.pdf) (Lee et al., ACL 2024 Findings)  
9. [HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models](https://arxiv.org/abs/2405.14831) (Jiménez Gutiérrez et al., NeurIPS 2024)  
10. [MemLong: Memory-Augmented Retrieval for Long Text Modeling](https://arxiv.org/abs/2408.16967) (Liu et al., arXiv 2024)  
11. [Memory Layers at Scale](https://arxiv.org/abs/2412.09764) (Berges et al., arXiv 2024)  
12. [M+: Extending MEMORYLLM with Scalable Long-Term Memory](https://arxiv.org/abs/2502.00592) (Wang et al., ICML 2025)  
13. [From RAG to Memory: Non-Parametric Continual Learning for Large Language Models](https://arxiv.org/abs/2502.14802) (Jiménez Gutiérrez et al., ICML 2025)  
14. [Can Memory-Augmented Language Models Generalize on Reasoning-in-a-Haystack Tasks?](https://arxiv.org/abs/2503.07903) (Das et al., arXiv 2025)  

#### Chat Assistant
1. [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) (Packer et al., 2023)  
2. [MemoryBank: Enhancing Large Language Models with Long-Term Memory](https://ojs.aaai.org/index.php/AAAI/article/view/29946) (Zhong et al., AAAI 2024)  
3. [Embracing Compressive Memory in Real-World Long-Term Conversations](https://arxiv.org/abs/2402.11975) (Liu et al., 2024)  
4. [Hello Again! LLM-powered Personalized Agent for Long-term Dialogue (LD-Agent)](https://arxiv.org/abs/2406.05925) (Li et al., 2024)  
5. [Self-evolving Personalized Dialogue Agents (SPDA)](https://arxiv.org/abs/2406.13960) (Cheng et al., 2024)  
6. [CarMem: Enhancing Long-Term Memory in LLM Voice Assistants through Category-Bounding](https://arxiv.org/abs/2501.09645) (Kirmayr et al., 2025)  
7. [SeCom: On Memory Construction and Retrieval for Personalized Conversational Agents](https://arxiv.org/abs/2502.05589) (Pan et al., ICLR 2025)  
8. [In Prospect and Retrospect: Reflective Memory Management for Long-Term Personalized Dialogue Agents](https://aclanthology.org/2025.acl-long.413/) (Tan et al., ACL 2025)  
9. [CAIM: Cognitive AI Memory Framework for Long-Term Interaction with Intelligent Agents](https://arxiv.org/abs/2505.13044) (Westhäußer et al., 2025)  
10. [Prime: LLM Personalization with Cognitive Memory and Thought Processes](https://arxiv.org/abs/2507.04607) (Zhang et al., 2025)  
11. [MemAgent: Reshaping Long-Context LLM with Multi-Conversation RL-Based Memory Agent](https://arxiv.org/abs/2507.02259) (Yu et al., 2025)  
12. [SGMem: Sentence Graph Memory for Long-Term Conversational Agents](https://arxiv.org/abs/2509.21212) (Zhang et al., 2025)

#### Reasoning
1. [Sleep-Time Compute: Beyond Inference Scaling at Test-Time](https://arxiv.org/abs/2504.13171) (Lin et al., arXiv 2025)  
2. [ReasonIR: Training Retrievers for Reasoning Tasks](https://arxiv.org/abs/2504.20595) (Shao et al., LM-Conf 2025)  
3. [Contextual Experience Replay for Self-Improvement of Language Agents](https://aclanthology.org/2025.acl-long.694/) (Liu et al., ACL 2025)  
4. [Emergent Hierarchical Reasoning in LLMs through Reinforcement Learning](https://arxiv.org/abs/2509.03646) (H. Wang et al., arXiv 2025)  

#### Agent & Multi-Modal Systems (incl. embodied agents)
1. [Synapse: Trajectory-as-Exemplar Prompting with Memory for Computer Control](https://arxiv.org/abs/2306.07863) (Zheng et al., ICLR 2024)  
2. [Agent Workflow Memory](https://arxiv.org/abs/2409.07429) (Z. Wang et al., arXiv 2024)  
3. [3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning](https://arxiv.org/abs/2411.17735) (Yang et al., arXiv 2024)  
4. [Embodied VideoAgent: Persistent Memory from Egocentric Videos and Embodied Sensors](https://arxiv.org/abs/2501.00358) (Fan et al., arXiv 2024)  
5. [3DLLM-Mem: Long-Term Spatial-Temporal Memory for Embodied 3D LLMs](https://arxiv.org/abs/2505.22657) (Hu et al., arXiv 2025)  
6. [MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents](https://arxiv.org/abs/2506.15841) (Zhou et al., arXiv 2025)  
7. [Reasoning and Planning for Long-term Active Embodied QA (LA-EQA)](https://arxiv.org/abs/2507.12846) (Ginting et al., arXiv 2025)  
8. [Agent KB: Leveraging Cross-Domain Experience for Agentic Problem Solving](https://arxiv.org/abs/2507.06229) (Tang et al., arXiv 2025)  
9. [MemTool: Optimizing Short-Term Memory Management for Dynamic Tool Calling in LLM Agents](https://arxiv.org/abs/2507.21428) (Lumer et al., arXiv 2025)  
10. [ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory](https://arxiv.org/abs/2509.25140) (Ouyang et al., arXiv 2025)  

### Benchmarks
* Needle-in-a-Haystack — [GitHub](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)  
* L-Eval — [arXiv](https://arxiv.org/abs/2307.11088), [OpenReview](https://openreview.net/forum?id=eUAr4HwU0X), [GitHub](https://github.com/OpenLMLab/LEval)  
* LongBench — [arXiv](https://arxiv.org/abs/2308.14508), [ACL Anthology](https://aclanthology.org/2024.acl-long.172/), [GitHub](https://github.com/THUDM/LongBench)  
* ∞Bench / InfiniteBench — [arXiv](https://arxiv.org/abs/2402.13718), [ACL Anthology](https://aclanthology.org/2024.acl-long.814/), [GitHub](https://github.com/OpenBMB/InfiniteBench)  
* PerLTQA — [arXiv](https://arxiv.org/abs/2402.16288), [ACL Anthology](https://aclanthology.org/2024.sighan-1.18/), [GitHub](https://github.com/Elvin-Yiming-Du/PerLTQA)  
* LoCoMo — [arXiv](https://arxiv.org/abs/2402.17753), [project](https://snap-research.github.io/locomo/), [ACL Anthology](https://aclanthology.org/2024.acl-long.747/)  
* RULER — [arXiv](https://arxiv.org/abs/2404.06654), [GitHub](https://github.com/NVIDIA/RULER), [OpenReview](https://openreview.net/forum?id=kIoBbc76Sy)  
* HELMET — [arXiv](https://arxiv.org/abs/2410.02694), [OpenReview](https://openreview.net/forum?id=293V3bJbmE), [GitHub](https://github.com/princeton-nlp/HELMET), [project](https://princeton-nlp.github.io/HELMET/)  
* LongMemEval — [arXiv](https://arxiv.org/abs/2410.10813), [GitHub](https://github.com/xiaowu0162/LongMemEval), [website](https://xiaowu0162.github.io/long-mem-eval/)  
* Episodic Memories Generation and Evaluation Benchmark — [arXiv](https://arxiv.org/abs/2501.13121)  
* PersonaMem — [arXiv](https://arxiv.org/abs/2504.14225), [project](https://zhuoqunhao.github.io/PersonaMem.github.io/), [GitHub](https://github.com/bowen-upenn/PersonaMem), [HF dataset](https://huggingface.co/datasets/bowen-upenn/PersonaMem)  
* 3DMem-Bench (from 3DLLM-Mem) — [arXiv](https://arxiv.org/abs/2505.22657), [project](https://3dllm-mem.github.io/)  
* MemBench — [arXiv](https://arxiv.org/abs/2506.21605), [ACL Anthology](https://aclanthology.org/2025.findings-acl.989.pdf)  
* BFCL V4 — [leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html), [memory blog](https://gorilla.cs.berkeley.edu/blogs/16_bfcl_v4_memory.html), [Gorilla repo](https://github.com/ShishirPatil/gorilla)

### Resources
* MemGPT (project page) — [Berkeley project](https://sky.cs.berkeley.edu/project/memgpt/)  
* Zep — [GitHub](https://github.com/getzep/zep), [Python SDK](https://github.com/getzep/zep-python), [Go SDK](https://github.com/getzep/zep-go)  
* LangChain / LangGraph memory — [LangMem GitHub](https://github.com/langchain-ai/langmem), [LangGraph memory docs](https://langchain-ai.github.io/langgraph/concepts/memory/)  
* LlamaIndex memory — [docs](https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/memory/)  
* Mem0 — [GitHub](https://github.com/mem0ai/mem0), [website](https://mem0.ai/)  
* Letta (formerly MemGPT) — [GitHub](https://github.com/letta-ai/letta), [website](https://www.letta.com/)  
