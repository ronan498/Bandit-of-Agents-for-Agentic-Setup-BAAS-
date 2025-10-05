# Bandit-of-Agents for Agentic Setup (BAAS)

**One-liner:** A contextual multi-armed bandit that selects among **agentic pipeline patterns** (arms) to set up an LLM-agent framework for a user’s environment; an **inner-loop tuner** adapts plan knobs per run. Rewards are grounded by lightweight validators.

## Why
Most agentic stacks use fixed pipelines. BAAS treats pipelines themselves as actions and **learns which pattern fits which environment** (Codespaces, local PC, GPU/CPU, libraries present, etc.).

## Architecture
- **Context probe:** `baas/contexts.py` → numeric feature vector (OS, Python, GPU, pkg managers, cloud hints, ML libs, budget flags).
- **Bandit (outer loop):** LinUCB over arms.
- **Arms:** `baas/arms/patterns.py` registers pipeline patterns and executes them.
- **Inner RL (micro):** `baas/inner_rl/reinforce.py` does REINFORCE-style tuning of a plan knob (e.g., `max_steps`).
- **Validators:** `baas/exec/validators.py` produce deterministic, environment-grounded rewards.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Toy simulation (synthetic users)
python experiments/run_sim.py

# Real-context demo (your env)
python experiments/run_demo_local.py