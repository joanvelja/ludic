# Ludic Domain Glossary

> Auto-generated documentation - Last updated: 2026-01-06

## Core Concepts

### Action
The string output from an agent that the environment processes.

### Agent
Stateful LLM wrapper with inference, context management, and parsing.

### Agent-Environment Loop (AEL)
The interaction between agent and environment, mediated by an InteractionProtocol.

### Attention Mask
Binary tensor: 1 for valid tokens, 0 for padding.

### Batch Source
Data source yielding SAWBatches to the Trainer.

### ChatClient
Protocol for inference backends. Implements `complete()` and `sync_weights()`.

### ChatResponse
Normalized LLM output: text, token_ids, logprobs, finish_reason.

### Context Strategy
Owns chat transcript and controls prompt construction.

### Credit Assignment
Algorithm computing per-step scalar weights from rollouts.

### Curriculum
Sequence of RolloutRequests defining what to learn.

### Environment
Provides state transitions: obs → action → (next_obs, reward).

### Episode
Complete trajectory from reset to termination/truncation.

### Finish Reason
LLM stop condition: "stop", "length", "function_call".

### GRPO
Group-wise Relative Policy Optimization: A_i = R_i - mean(R_group).

### Interaction Protocol
Defines agent-environment interaction rules.

### Loss Function
Maps logits + batch → scalar loss.

### Parser
Decodes LLM text into action. Returns ParseResult(action, reward, obs).

### Policy
Learned mapping from observations to actions (the LLM weights).

### Policy Publisher
Interface to push weights to inference workers.

### Return-to-Go
Discounted cumulative future reward: G_t = Σ γ^k r_{t+k}.

### Rollout
Complete trajectory of Steps representing one episode.

### Rollout Engine
Stateless factory generating rollouts from RolloutRequests.

### SAWBatch
Collated batch of SAWItems for training.

### SAWItem (State-Action-Weight)
Training sample: state_ids, action_ids, masks, weight, metadata.

### Step
Single transition: (prev_obs, action, next_obs, reward, flags).

### StepOutcome
Environment's response: obs, reward, truncated, terminated, info.

### Terminated
True if episode reached natural terminal state.

### Truncated
True if episode cut short (max_steps, timeout).

## Algorithm Names

| Acronym | Full Name |
|---------|-----------|
| CISPO | Clipped IS-weight Policy Optimization |
| DPO | Direct Preference Optimization |
| FSDP | Fully Sharded Data Parallel |
| GRPO | Group-wise Relative Policy Optimization |
| KL | Kullback-Leibler divergence |
| NCCL | NVIDIA Collective Communications Library |
| PPO | Proximal Policy Optimization |
| ReAct | Reasoning + Acting |
| REINFORCE | Policy gradient algorithm |
| RL | Reinforcement Learning |
| RLHF | RL from Human Feedback |
| SAW | State-Action-Weight |
| ScaleRL | Sample-efficient RL recipe |
| SFT | Supervised Fine-Tuning |
| ZVP | Zero-Variance Prompt |

## File Locations

| Concept | File |
|---------|------|
| Core types | `src/ludic/types.py` |
| LudicEnv | `src/ludic/envs/env.py` |
| SingleAgentEnv | `src/ludic/envs/single_agent_env.py` |
| Agent | `src/ludic/agents/base_agent.py` |
| ContextStrategy | `src/ludic/context/base.py` |
| Parsers | `src/ludic/parsers.py` |
| InteractionProtocol | `src/ludic/interaction/base.py` |
| RolloutEngine | `src/ludic/training/batching/rollout_engine.py` |
| CreditAssigner | `src/ludic/training/credit_assignment.py` |
| Loss functions | `src/ludic/training/loss.py` |
| RLAlgorithm | `src/ludic/training/algorithm.py` |
| Trainer | `src/ludic/training/trainer.py` |
| ChatClient | `src/ludic/inference/client.py` |
| VLLMChatClient | `src/ludic/inference/vllm_client.py` |
