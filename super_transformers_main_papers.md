Recursive Transformers

Hierarchical Reasoning Model (HRM)

Inspired by brain's hierarchical processing, HRM delivers unprecedented reasoning power on complex tasks like ARC-AGI and expert-level Sudoku using just 1k examples, no pretraining or CoT!

https://arxiv.org/abs/2506.21734

Abstract: Reasoning, the process of devising and executing complex goal-oriented action sequences, remains a critical challenge in AI. Current large language models (LLMs) primarily employ Chain-of-Thought (CoT) techniques, which suffer from brittle task decomposition, extensive data requirements, and high latency. Inspired by the hierarchical and multi-timescale processing in the human brain, we propose the Hierarchical Reasoning Model (HRM), a novel recurrent architecture that attains significant computational depth while maintaining both training stability and efficiency. HRM executes sequential reasoning tasks in a single forward pass without explicit supervision of the intermediate process, through two interdependent recurrent modules: a high-level module responsible for slow, abstract planning, and a low-level module handling rapid, detailed computations. With only 27 million parameters, HRM achieves exceptional performance on complex reasoning tasks using only 1000 training samples. The model operates without pre-training or CoT data, yet achieves nearly perfect performance on challenging tasks including complex Sudoku puzzles and optimal path finding in large mazes. Furthermore, HRM outperforms much larger models with significantly longer context windows on the Abstraction and Reasoning Corpus (ARC), a key benchmark for measuring artificial general intelligence capabilities. These results underscore HRM's potential as a transformative advancement toward universal computation and general-purpose reasoning systems.



Tiny Recursive Model (TRM)

Samsung’s Tiny Recursive Model (TRM) is about 10,000x smaller than typical LLMs yet smarter because it thinks recursively instead of just predicting text. It first drafts an answer, then builds a hidden "scratchpad" for reasoning, repeatedly critiques and refines its logic (up to 16 times), and produces improved answers each cycle.

https://arxiv.org/abs/2510.04871

Abstract: Hierarchical Reasoning Model (HRM) is a novel approach using two small neural networks recursing at different frequencies. This biologically inspired method beats Large Language models (LLMs) on hard puzzle tasks such as Sudoku, Maze, and ARC-AGI while trained with small models (27M parameters) on small data (around 1000 examples). HRM holds great promise for solving hard problems with small networks, but it is not yet well understood and may be suboptimal. We propose Tiny Recursive Model (TRM), a much simpler recursive reasoning approach that achieves significantly higher generalization than HRM, while using a single tiny network with only 2 layers. With only 7M parameters, TRM obtains 45% test-accuracy on ARC-AGI-1 and 8% on ARC-AGI-2, higher than most LLMs (e.g., Deepseek R1, o3-mini, Gemini 2.5 Pro) with less than 0.01% of the parameters.



Probabilistic Tiny Recursive Model

https://arxiv.org/abs/2605.19943

Abstract: Tiny Recursive Models (TRM) solve complex reasoning tasks with a fraction of the parameters of modern large language models (LLMs) by iteratively refining a latent state and final answer. While powerful, their deterministic recursion can lead to convergence at suboptimal solutions, without escape mechanism. A common workaround relies on task-specific input perturbations at test time combined with answer aggregation via voting. We introduce Probabilistic TRM (PTRM), a task-agnostic framework for test-time compute scaling that addresses this limitation through stochastic exploration. PTRM injects Gaussian noise at each deep recursion step, enabling parallel trajectories to explore diverse solution basins, and selects among them using the model's existing Q head (used for early stopping in the original TRM). Without requiring retraining or task-specific augmentations, PTRM enables substantial accuracy gains across benchmarks, including Sudoku-Extreme (87.4% to 98.75%) and on various puzzles from Pencil Puzzle Bench (62.6% to 91.2%). On the latter, PTRM achieves nearly double the accuracy of frontier LLMs (91.2% vs. 55.1%) at less than 0.0001x the cost, using only 7M parameters.



Recursive Language Model (RLM)

What if scaling the context windows of frontier LLMs is much easier than it sounds? We’re excited to share our work on Recursive Language Models (RLMs). A new inference strategy where LLMs can decompose and recursively interact with input prompts of seemingly unbounded length, as a REPL environment. On the OOLONG benchmark, RLMs with GPT-5-mini outperforms GPT-5 by over 110% gains (more than double!) on 132k-token sequences and is cheaper to query on average. On the BrowseComp-Plus benchmark, RLMs with GPT-5 can take in 10M+ tokens as their “prompt” and answer highly compositional queries without degradation and even better than explicit indexing/retrieval

https://alexzhang13.github.io/blog/2025/rlm/

https://towardsdatascience.com/recursive-language-models-one-example-deep-dive-that-explains-everything/

https://arxiv.org/abs/2512.24601

Abstract: We study allowing large language models (LLMs) to process arbitrarily long prompts through the lens of inference-time scaling. We propose Recursive Language Models (RLMs), a general inference paradigm that treats long prompts as part of an external environment and allows the LLM to programmatically examine, decompose, and recursively call itself over snippets of the prompt. We find that RLMs can successfully process inputs up to two orders of magnitude beyond model context windows and, even for shorter prompts, dramatically outperform the quality of vanilla frontier LLMs and common long-context and coding scaffolds (e.g., on GPT-5 by a median across the evaluated benchmarks of 26% against compaction, 130% against CodeAct with sub-calls, and 13% against Claude Code) across four diverse long-context tasks while having comparable cost. At a small scale, we post-train the first model around the RLM. Our model, RLM-Qwen3-8B, outperforms the underlying Qwen3-8B model by 28.3% on average and even approaches the quality of vanilla GPT-5 on three long-context tasks. Code is available here: https://github.com/alexzhang13/rlm



lambda-RLM

Very interesting variation on RLM. A lot more of the work is handled by deterministic recursive decomposition rather than the learned policy of the LLM.

https://arxiv.org/abs/2603.20105

Abstract: LLMs are increasingly used as general-purpose reasoners, but long inputs remain bottlenecked by a fixed context window. Recursive Language Models (RLMs) address this by externalising the prompt and recursively solving subproblems. Yet existing RLMs depend on an open-ended read-eval-print loop (REPL) in which the model generates arbitrary control code, making execution difficult to verify, predict, and analyse.
We introduce λ-RLM, a framework for long-context reasoning that replaces free-form recursive code generation with a typed functional runtime grounded in λ-calculus. It executes a compact library of pre-verified combinators and uses neural inference only on bounded leaf subproblems, turning recursive reasoning into a structured functional program with explicit control flow. We show that λ-RLM admits formal guarantees absent from standard RLMs, including termination, closed-form cost bounds, controlled accuracy scaling with recursion depth, and an optimal partition rule under a simple cost model. Empirically, across four long-context reasoning tasks and nine base models, λ-RLM outperforms standard RLM in 29 of 36 model-task comparisons, improves average accuracy by up to +21.9 points across model tiers, and reduces latency by up to 4.1x. These results show that typed symbolic control yields a more reliable and efficient foundation for long-context reasoning than open-ended recursive code generation. The complete implementation of λ-RLM, is open-sourced for the community at: https://github.com/lambda-calculus-LLM/lambda-RLM



Tufa Labs, "The Duck"

Winning Solution for ARC-AGI-3 Milestone 1. A small open-source LLM that plays ARC-AGI-3 by writing and running Python in a live REPL (similar to the Duke Harness), treating each game like an interactive coding problem.

What's interesting:

This submission is the only one of the winners that has the "agent-writes-code" approach. The other two simply use "pick a JSON action."
It is a spiritual successor to "Stochastic Goose," which won the earlier ARC-AGI-3 Agent Preview Competition.
Runs Qwen 3.6 27B FP8 locally.
Tufa Labs noted that, counterintuitively, hand-crafted tools actually hurt the model; letting it improvise worked better.
Featured on Machine Learning Street Talk.

Notebook: https://www.kaggle.com/code/jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner

Write-up: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133

Video: https://www.youtube.com/watch?v=Vg6FBKTlfOw



Looped Transfomer / Looped Language Model (LoopLM)

Scaling Latent Reasoning via Looped Language Models

https://arxiv.org/abs/2510.25741

Abstract: Modern LLMs are trained to "think" primarily via explicit text generation, such as chain-of-thought (CoT), which defers reasoning to post-training and under-leverages pre-training data. We present and open-source Ouro, named after the recursive Ouroboros, a family of pre-trained Looped Language Models (LoopLM) that instead build reasoning into the pre-training phase through (i) iterative computation in latent space, (ii) an entropy-regularized objective for learned depth allocation, and (iii) scaling to 7.7T tokens. Ouro 1.4B and 2.6B models enjoy superior performance that match the results of up to 12B SOTA LLMs across a wide range of benchmarks. Through controlled experiments, we show this advantage stems not from increased knowledge capacity, but from superior knowledge manipulation capabilities. We also show that LoopLM yields reasoning traces more aligned with final outputs than explicit CoT. We hope our results show the potential of LoopLM as a novel scaling direction in the reasoning era. Our model is available here: https://ouro-llm.github.io/



Generative Recursive Reasoning

Recursive Reasoning Models like HRM, TRM, and Looped Transformers are deterministic — same input, same reasoning, every time. They collapse the entire space of plausible reasoning paths into a single attractor.

Our model GRAM (Generative Recursive reAsoning Models) turns recursion itself into a stochastic latent trajectory. Multiple hypotheses, alternative solution strategies, and inference-time scaling not just by depth, but by width — parallel trajectory sampling.

And here's the kicker: the same formulation that gives us conditional reasoning p(y|x) also makes GRAM a general generative model p(x).

With only 10M params:
• Sudoku-Extreme: 97.0% (TRM 87.4%)
• ARC-AGI-1: 52.0%
• ARC-AGI-2: 11.1%
• N-Queens coverage: 90%+

https://ahn-ml.github.io/gram-website/

https://arxiv.org/abs/2605.19376

Abstract: How should future neural reasoning systems implement extended computation? Recursive Reasoning Models (RRMs) offer a promising alternative to autoregressive sequence extension by performing iterative latent-state refinement with shared transition functions. Yet existing RRMs are largely deterministic, following a single latent trajectory and converging to a single prediction. We introduce Generative Recursive reAsoning Models (GRAM), a framework that turns recursive latent reasoning into probabilistic multi-trajectory computation. GRAM models reasoning as a stochastic latent trajectory, enabling multiple hypotheses, alternative solution strategies, and inference-time scaling through both recursive depth and parallel trajectory sampling. This yields a latent-variable generative model supporting conditional reasoning via pθ(y∣x) and, with fixed or absent inputs, unconditional generation via pθ(x). Trained with amortized variational inference, GRAM improves over deterministic recurrent and recursive baselines on structured reasoning and multi-solution constraint satisfaction tasks, while demonstrating an unconditional generation capability. https://ahn-ml.github.io/gram-website/


Tooling and Workflows Enhancing LLMs

LimiX: Unleashing Structured-Data Modeling Capability for Generalist Intelligence

Paper: https://arxiv.org/abs/2509.03505
Project: https://limix.ai
Code: https://github.com/limix-ldm/LimiX
Huggingface: https://huggingface.co/stable-ai

Abstract: We argue that progress toward general intelligence requires complementary foundation models grounded in language, the physical world, and structured data. This report presents LimiX-16M and LimiX-2M, two instantiations of our large structured-data models (LDMs). Both models treat structured data as a joint distribution over variables and missingness, thus capable of addressing a wide range of tabular tasks through query-based conditional prediction via a single model. They are pretrained using masked joint-distribution modeling with an episodic, context-conditional objective, supporting rapid, training-free adaptation at inference. We evaluate LimiX models across 11 large structured-data benchmarks with broad regimes of sample size, feature dimensionality, class number, categorical-to-numerical feature ratio, missingness, and sample-to-feature ratios. LimiX-16M consistently surpasses strong baselines, as shown in Figure 1 and Figure 2. The superiority holds across a wide range of tasks, such as classification, regression, missing value imputation, and data generation, often by substantial margins, while avoiding task-specific architectures or bespoke training per task. Notably, LimiX-2M delivers strong results under tight compute and memory budgets. We also present the first scaling law study for LDMs, revealing how data and model scaling jointly influence downstream performance and offering quantitative guidance for tabular foundation modeling. All LimiX models are publicly accessible under Apache 2.0.



Solving a Million-Step LLM Task with Zero Errors

https://arxiv.org/abs/2511.09030

Abstract: LLMs have achieved remarkable breakthroughs in reasoning, insights, and tool use, but chaining these abilities into extended processes at the scale of those routinely executed by humans, organizations, and societies has remained out of reach. The models have a persistent error rate that prevents scale-up: for instance, recent experiments in the Towers of Hanoi benchmark domain showed that the process inevitably becomes derailed after at most a few hundred steps. Thus, although LLM research is often still benchmarked on tasks with relatively few dependent logical steps, there is increasing attention on the ability (or inability) of LLMs to perform long range tasks. This paper describes MAKER, the first system that successfully solves a task with over one million LLM steps with zero errors, and, in principle, scales far beyond this level. The approach relies on an extreme decomposition of a task into subtasks, each of which can be tackled by focused microagents. The high level of modularity resulting from the decomposition allows error correction to be applied at each step through an efficient multi-agent voting scheme. This combination of extreme decomposition and error correction makes scaling possible. Thus, the results suggest that instead of relying on continual improvement of current LLMs, massively decomposed agentic processes (MDAPs) may provide a way to efficiently solve problems at the level of organizations and societies.



Transformers as Computers

Can LLMs Be Computers? Executing programs inside transformers with exponentially faster inference

https://www.percepta.ai/blog/can-llms-be-computers

Language models can solve tough math problems at research grade but struggle on simple computational tasks that involve reasoning over many steps and long context. Even multiplying two numbers or solving small Sudokus is nearly impossible unless they rely on external tools.

But what does it take for an LLM itself to be as reliable and efficient as a computer?

We answer this by literally building a computer inside a transformer. We turn arbitrary C code into tokens that the model itself can execute reliably for millions of steps in seconds.



Neural Computers

https://arxiv.org/abs/2604.06425

Abstract: We propose a new frontier: Neural Computers (NCs) that unify computation, memory, and I/O of traditional computers in a learned runtime state. Our long-term goal is the Completely Neural Computer (CNC): the mature, general-purpose realization of this emerging machine form, with stable execution, explicit reprogramming, and durable capability reuse. As an initial step, we study whether elementary NC primitives can be learned solely from collected I/O traces, without instrumented program state. Concretely, we instantiate NCs as video models that roll out screen frames from instructions, pixels, and user actions (when available) in CLI and GUI settings. We show that NCs can acquire elementary interface primitives, especially I/O alignment and short-horizon control, while routine reuse, controlled updates, and symbolic stability remain challenging. We outline a roadmap toward CNCs, to establish a new computing paradigm beyond today's agents and conventional computers.



Loop the Loopies!

https://arxiv.org/abs/2607.16051

Full-PDF mechanism summary (v2, 38 pages): Loopie changes the order of tied computation. A conventional model-loop applies the stored stack and then repeats it (`B1...BL,B1...BL`); its fixed `R=2` layer-loop applies each stored Attention/MoE block twice before advancing (`B1,B1,B2,B2,...`). The paper couples this schedule with activation checkpointing, a larger microbatch, and reinvestment of measured training efficiency into width/depth. Its selected model is not FLOP-matched to the reference: the paper's work proxy is about 1.424x, while nominal measured TFLOPS rise from 189.65 to 261.53 in the reported setup. Source ablations suggest layer-loop overtakes model-loop late in pretraining and that `R=2` offers the best marginal tradeoff among tested recurrence counts, but seeds, confidence intervals, hardware details, inference compute, and many causal ablations are missing.

The layer-loop schedule matches Mixture-of-Recursions' Sequence-sharing layout, but the model uses only ordinary MoE expert routing and has no MoR recursion-depth router, learned exit, loop embedding, visit adapter, or entropy depth allocator. Its supervised post-training uses target-only loss masking, but the authors acknowledge incomplete ablation; that is a separate conditional analogue rather than evidence for layer-loop. No SAIR reproduction exists. The project cross-registers the schedule at `ST-M009.a`/`ST-M043.a`, maps capacity recycling to `ST-M043`, target-only continuation to conditional `ST-M044`, and a dense-first blocked preregistration to EXP-655. The frozen PDF is `research/papers/arxiv-2607.16051.pdf`, SHA-256 `1abfc7644b0253119b53b86327011471a0d017c2904d648461fc9d6a10b074ce`.
