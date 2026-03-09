# Future Directions: Autonomous AI Research at Scale

*Thoughts after running ~1,060 parallel experiments overnight across 8 H100 GPUs — March 2026*

---

## 1. Recursive Self-Improvement

The ultimate goal of this paradigm is for LLMs to recursively improve themselves. The current loop is:

```
Human sets goal → Agent experiments → Human reviews → repeat
```

The vision is:

```
Human sets high-level direction → Agents experiment, generate hypotheses,
critique each other, and improve the research process itself → repeat
```

The human bottleneck of annotation, labeling, and evaluation (the Scale AI model) disappears. Agents generate their own training signal by running experiments and measuring outcomes. The research org code (`program.md`) is itself subject to optimization — agents should be able to propose better versions of their own instructions.

This isn't science fiction: we're already one step away. The missing piece is agents that can modify `program.md` and measure whether downstream experiments improve.

---

## 2. Literature-Aware Ideation

Current agents propose ideas from their training distribution plus context. A stronger setup:

**Before generating a hypothesis**, the agent:
1. Reads all relevant papers (Chinchilla, Muon, NorMuon, RetNet, FlashAttention, etc.)
2. Reads relevant GitHub repos and open issues
3. Reads community discussions (Twitter/X threads, Hacker News, Reddit ML)
4. Reasons explicitly about what's **underexplored** — not just what's known to work

This shifts the agent from *reproducing known good ideas* to *filling gaps in the literature*. The agent's ideation is grounded in the actual frontier of human knowledge, not just its pre-training snapshot.

Practically: pipe arxiv RSS + GitHub trending + `program.md` into context before each hypothesis generation step.

---

## 3. Multi-Agent Debate

Current setup: 8 independent agents, zero coordination.

Better: **propose → critique → refine** before committing to an experiment.

```
Agent A: "I propose trying linear attention to reduce memory bandwidth."
Agent B: "Linear attention degrades on long-range dependencies. Have you considered—"
Agent A: "Fair. What if we only apply it to lower layers?"
[consensus or veto]
→ Experiment runs
```

This is a form of "AI peer review" before incurring the cost of a 5-minute training run. The expected value of each experiment increases; crash rate and obviously-bad ideas decrease.

MAP-Elites (already used here) handles behavioral diversity across the population. Debate handles quality within each individual proposal. These are complementary.

---

## 4. Reproduction-Driven Discovery

A powerful, underrated research strategy: **have AI reproduce all existing published work**.

The reproduction process itself produces information that doesn't appear in papers:
- Which hyperparameters matter most (sensitivity analysis, for free)
- Which results are fragile vs. robust
- Which baselines were actually weaker than claimed
- Implementation details that the paper glosses over

Example: reproducing the Muon optimizer paper from scratch likely reveals that `ns_steps=5` is somewhat arbitrary, or that the optimal `beta2` is hardware-dependent. These are things the paper doesn't say.

Concretely: maintain a "reproduction queue" — every week, add 3 recent ML papers. Assign one agent to reproduce each. Log discrepancies between claimed and achieved results. The discrepancy log is itself valuable training data.

---

## 5. Continuous Operation with Diversity Maintenance

The 8-GPU overnight run showed that agents converge: by hour 6, all 8 GPUs had independently found batch=2^18 and AR=96. This is good for confidence but bad for continued exploration.

The problem is analogous to **genetic drift** in evolutionary computation: a population that all exploit the same optimum stops producing useful variation.

Solutions from evolutionary computation:

- **MAP-Elites** (already in use): maintain a behavioral archive, reward exploration of new cells regardless of fitness improvement
- **Fitness sharing / niching**: penalize experiments that are too similar to recent experiments (cosine similarity of hyperparameter vectors)
- **Explicit diversity injection**: periodically force one agent to explore a direction that hasn't been tried in the last N experiments
- **Restart with memory**: when an agent's best val_bpb hasn't improved in K experiments, restart from a random corner of the unexplored space (while keeping the archive)

The deeper point: **long-term autonomous research requires active diversity maintenance, not just hill-climbing**. Without it, the population collapses to a local optimum and you're paying for 8 GPUs to confirm the same thing repeatedly.

---

## 6. Summary

| Direction | Difficulty | Expected Impact |
|-----------|-----------|-----------------|
| Recursive self-improvement (agents improve `program.md`) | Medium | Very high |
| Literature-aware ideation | Low | High |
| Multi-agent debate before experiments | Medium | Medium-high |
| Reproduction-driven discovery | Low | Medium |
| Diversity maintenance (MAP-Elites extensions) | Low-medium | High for long runs |

The near-term highest-leverage improvement is **literature-aware ideation** (read papers before proposing) combined with **agents improving their own `program.md`**. These two together close the loop and make the system genuinely self-improving rather than just self-experimenting.

---

*Written after the March 8–9 2026 overnight run. The system achieved val_bpb 0.9723 from baseline 0.9979 in ~8 hours. See `docs/report_mar9.md` for full results.*
