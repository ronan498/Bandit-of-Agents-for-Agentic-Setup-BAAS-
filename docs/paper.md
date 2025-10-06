# Bandit-of-Agents: Contextual Bandits for Adaptive Agentic Pipeline Selection with LLM-Generated Arms

**Authors:** Your Name (Affiliation)

## Abstract
We present BAAS, a contextual multi-armed bandit that selects among agentic pipeline patterns (arms) to set up LLM-agent frameworks tailored to a user’s environment. BAAS builds a feature vector from the runtime context (OS, Python, GPU, package managers, ML stack) and uses LinUCB to pick a pipeline. A tiny inner-loop policy-gradient tuner adapts plan knobs (e.g., max steps). Rewards are grounded by lightweight validators that run deterministically without side effects. We demonstrate strong gains over non-contextual baselines and show extensibility with an LLM-planner arm and a graph-planner arm.

## 1. Introduction
Agentic AI stacks (planner→tools→reflect→retry) often assume a fixed pipeline. But setups vary widely across machines and cloud environments. We ask: **can the system learn which pipeline pattern fits a given environment, on the fly?**  
BAAS treats **pipelines as actions** and uses **contextual bandits** to select them, with a small inner tuner to adapt per-run.

**Contributions.**
1. A simple, practical **context→bandit→arm** loop that grounds rewards via deterministic validators.
2. **Inner-loop tuning** (REINFORCE) over plan knobs for per-run adaptation.
3. **LLM-generated arms** (with safe fallbacks) and a **graph-planner arm**, showing extensibility.
4. **Ablations**: contextual vs. non-contextual policies; tuner/no-tuner; arm diversity.

## 2. Related Work (brief)
- **Contextual bandits** (LinUCB, Thompson sampling): online decision-making with side information.  
- **AutoML / meta-controllers**: learning to choose pipelines/components.  
- **Agentic LLM frameworks**: planner-executor, tool routing, reflection/retry.  
- **LLM architecture search / program synthesis**: proposing new components under constraints.

## 3. Method

### 3.1 Context Vector
We probe the environment and form a numeric vector: OS one-hot, Python version, GPU/CUDA, package managers, cloud hints, ML stack presence, budget flags.

### 3.2 Arms (Pipeline Patterns)
Each arm defines:
- `plan_fn(ctx) → plan`: framework, tools, max steps
- `exec_fn(plan) → reward`: executes deterministic validators (CPU micro-test, step budget smoke, framework hints) → `reward ∈ [0,1]`.

Arms implemented:
- **basic_agentic** (LangChain/base),
- **reflective_agentic** (AutoGen/custom + reflect/retry),
- **llm_agentic** (LLM proposes JSON plan; safe fallback if no key),
- **graph_agentic** (router→tool→reflect→retry, graph-style).

### 3.3 Outer Policy: LinUCB
Given context vector `x` and arms `a`, LinUCB selects `argmax_a μ_a(x) + α·σ_a(x)`, balances exploration vs. exploitation, and updates per arm with observed reward.

### 3.4 Inner Tuner (REINFORCE)
Before execution we tune a single knob (e.g., `max_steps ∈ {3,5}`) using a tiny policy-gradient update with a moving baseline.

### 3.5 Reward Design
Deterministic validators (no installs, no network):
- **CPU micro-test** time and sanity,
- **smoke pass** if steps budget ≥ threshold,
- **framework hints** (tiny bonuses),
- clamp into `[0,1]`.  
This grounds learning without side effects; richer validators are future work.

## 4. Experiments

### 4.1 Synthetic Users (Contextual Bandits)
We compare **LinUCB (contextual)** vs. **ε-greedy** and **fixed-best-after-warmup** (non-contextual) on synthetic users where each arm has hidden linear weights.

**Figure 1. Cumulative regret.**  
Saved by the script at `experiments/fig_regret.png`.

**Tail per-step regret (last 10%):**
- LinUCB (contextual): ~0.037  
- ε-greedy (non-contextual): ~0.450  
- Fixed best after warmup: ~0.442

### 4.2 Real-Context Demo (Your Machine)
We run BAAS on the host environment; the bandit learns among {basic, reflective, LLM, graph} arms with a micro-tuner.

**Example output:** see `experiments/run_demo_local.py`.

## 5. Ablations & Analyses (planned)
- **No-context ablation:** replace LinUCB with non-contextual policies.
- **No-tuner ablation:** disable inner RL.
- **Arm diversity:** remove/ add arms; measure shifts.
- **Regret spikes → arm generation:** underperforming contexts trigger LLM proposals, gated by validators.

## 6. Limitations
- Reward is minimalistic; richer validators (install dry-runs, prompt unit tests, cost/time) required for production.
- Current arms are light abstractions; real frameworks may require side-effectful setup.
- Security & privacy considerations when probing context or calling LLMs.

## 7. Broader Impact
Adaptive pipeline selection can reduce setup friction for practitioners, but care is needed to avoid biased “preference” heuristics and to protect keys and system information.

## 8. Reproducibility
- Code: repo contains `experiments/ablation_regret.py` and `experiments/run_demo_local.py`.
- Seeded runs for synthetic users; deterministic validators.
- Figure outputs saved under `experiments/`.

## 9. Conclusion
BAAS shows that contextual bandits + tiny inner tuning can select agentic pipelines that fit the environment and improve over non-contextual baselines. Extending rewards and enabling LLM-generated arm growth are promising next steps.

## Appendix A. Commands to Reproduce
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)
python experiments/ablation_regret.py
python experiments/run_demo_local.py