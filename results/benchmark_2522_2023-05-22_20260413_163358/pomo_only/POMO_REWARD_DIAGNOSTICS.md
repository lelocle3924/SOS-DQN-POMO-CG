# POMO Reward Diagnostics

**Generated automatically** because the DP fallback ratio exceeded 30 %.

## Statistics
| Metric | Value |
|--------|-------|
| Total pricing calls | 22 |
| POMO successes | 15 |
| DP fallback calls | 7 |
| DP fallback ratio | 31.8% |

## Failure Mechanism Analysis

The high DP fallback ratio indicates that POMO is consistently failing to
find negative reduced cost columns within 3 sampling attempts.  Possible
root causes:

1. **Reward–cost misalignment during training.**
   If the REINFORCE reward used during pre-training does not match the
   exact cost formula `FixedCost + CostPerKm * dist + CostPerHour * time`,
   the learned policy optimises for the wrong objective.  The model may
   confidently produce routes that are sub-optimal under the real metric.

2. **Lack of explicit edge-feature attention.**
   The current Attention Model processes node features only.  Real-world
   edge costs (from the OSRM matrix) are encoded *implicitly* through the
   REINFORCE reward signal.  When the topology is highly non-Euclidean
   (e.g. one-way streets, highways) the model cannot distinguish cheap
   from expensive arcs without an edge-aware encoder layer.

3. **Covariate shift in dual values.**
   During training, dual values are synthetic and uniformly distributed.
   During CG inference, duals evolve with the LP relaxation and may
   concentrate in narrow ranges or spike for hard-to-cover customers.
   Normalisation (max-scaling to [0, 1]) mitigates this but does not
   eliminate it entirely.

## Proposed Improvements

- **Short term**: retrain POMO on sub-graphs sampled from the real
  distance matrix (Phase 0 fix) so the reward reflects true edge costs.
- **Medium term**: add a Graph Attention Network (GAT) encoder layer
  that takes the distance / travel-time matrix as edge features.
  This gives the model direct access to arc costs at every attention
  step, removing the need to learn them purely from reward.
- **Long term**: augment training with an auxiliary loss that predicts
  the reduced cost of each rollout (supervised signal from the DP
  solver), creating a hybrid RL + supervised approach.
