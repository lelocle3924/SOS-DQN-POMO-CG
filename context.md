Here is the updated chronological summary, reflecting the recent hardware migration for training while preserving the constraints for local testing.

## **Project Context & Goal**

**Objective:** Solve a Site-Dependent Vehicle Routing Problem with Time Windows (SD-VRPTW) for CEL Consulting (avg. 200 customers) as part of Prof Hop's SOS lab project.
**Methodology:** Combine classical Operations Research (Branch & Price) with Deep Reinforcement Learning (DRL).

* **POMO-CG (Attention Model):** Solves the pricing subproblem to generate high-quality candidate columns (routes) with negative reduced costs.
* **DQN-CG (Graph Neural Network):** Filters the generated candidate columns to select the optimal subset that minimizes total Column Generation (CG) iterations.

### Additional context

#### Feillet's Branching Rule for Branch-and-Price

In the context of the Vehicle Routing Problem with Time Windows (VRPTW), branching directly on the route selection variables ($\theta_k$) is generally avoided. While setting a route variable to 1 is straightforward, enforcing $\theta_k = 0$ is problematic because it restricts the subproblem from generating that specific path, which is difficult to implement efficiently and creates a highly unbalanced search tree.

Instead, Feillet describes a standard branching rule based on the original flow variables associated with the arcs in the network. The algorithm selects an arc $(v_i, v_j)$ that has a fractional flow value $f_{ij}$ strictly between 0 and 1. The flow $f_{ij}$ is calculated by summing the fractional values of all selected routes that traverse this specific arc.

From this selected arc, two branches are created:

- **The arc is forbidden ($f_{ij} = 0$):** In the Master Problem, any existing columns that utilize arc $(v_i, v_j)$ are removed. In the subproblem, the arc $(v_i, v_j)$ is deleted from the network to ensure no future columns are generated using it.


- **The arc is enforced ($f_{ij} = 1$):** Because each customer is visited exactly once, enforcing the path from $v_i$ to $v_j$ means that $v_i$ can only be followed by $v_j$. Therefore, all other outbound arcs from $v_i$ and all other inbound arcs to $v_j$ must be forbidden. Existing columns in the Master Problem that use these newly forbidden arcs are removed.

This arc-based branching rule guarantees that if all arc flows $f_{ij}$ are integer values (0 or 1), the resulting solution represents a feasible set of independent vehicle routes. Additionally, if the total number of vehicles used is fractional, an initial branching rule is often applied first to adjust the upper and lower limits of the fleet size constraint before branching on specific arcs.
---

### **Phase I: Architectural Critique & The Pivot**

* **Problem:** The initial design misunderstood Branch & Price branching constraints and ignored the symmetry-breaking nature of Site Dependency. The initial codebase had fatal bugs (treating LP relaxations as integers, formulating Set Partitioning instead of Set Covering).
* **Solution:** Scrapped the codebase and established a rigorous new architecture.
* *B&P Strategy:* Branch on edges (Feillet’s rule), not routes.
* *RMP Strategy:* Formulate Master Problem as a Set Covering problem (inequality) using OR-Tools (GLOP).
* *Pricing Strategy:* Orchestrate $K$ separate POMO runs for each vehicle type to handle Site Dependency.



### **Phase II: Core Plumbing & The "Micro-Batch" Proof**

* **Problem:** Initial testing and development were restricted to a local machine (CPU, 8GB RAM), requiring extreme memory caution.
* **Solution:** Built the basic CG plumbing for the Root Node only. Implemented the POMO training loop using REINFORCE with a multi-start baseline. Executed a "Micro-Batch Overfit" test on a static 10-node graph.
* **Outcome:** The test perfectly converged, mathematically proving the complex transition masking and Attention logic worked before touching a GPU.

### **Phase III: Scaling & Infrastructure Hardening**

* **Design Concept:** Curriculum Learning (50 -> 100 -> 200 customers) designed for headless execution.
* **Problems Raised & Solved:**
* *Session Disconnects:* Implemented robust stateful checkpointing (saving model, Adam optimizer momentum, RNG states, and `best_val`).
* *Config Amnesia:* Saved active config overrides to the run folder so resumes don't revert to default parameters.
* *CUDA RNG Crash:* Fixed a serialization crash by mapping the RNG state to `.cpu().byte()` upon load.
* *Resumption Dip:* Diagnosed a temporary loss spike upon resuming as normal Adam momentum clashing with new RNG sequences.



### **Phase IV: Algorithmic Scrutiny & Math Fixes**

During rigorous code reviews, we identified and patched several critical bugs:

* **Float32 Drift:** NN `float32` accumulation caused precision drift in exact reduced cost. *Fix:* Decoupled OR math; recalculated costs from scratch using `float64` matrices.
* **Missing Fleet Duals:** Added fleet size constraints to the Master Problem and subtracted the corresponding depot dual from the reduced cost.
* **Advantage Explosion:** The $-1e6$ penalty for invalid routes poisoned the POMO baseline, causing NaN losses. *Fix:* Masked the baseline calculation to ignore heavily penalized rollouts.
* **Empty Route Degeneracy:** The "depot escape hatch" created zero-cost empty columns. *Fix:* Orchestrator strictly drops empty routes.

### **Phase V: Inference Testing & Fundamental Re-engineering**

At Epoch 200, we tested root node inference. The CG loop worked, but the POMO agent's behavior revealed deep structural flaws:

* **Problem 1: Safety Bias.** The agent learned to immediately return to the depot to avoid the $-1e6$ time-window penalty.
* *Solution:* **Proactive Look-Ahead Masking.** Removed the reactive penalty entirely. The environment now pre-calculates if a vehicle can safely return; if not, that customer is masked out mathematically.


* **Problem 2: Covariate Shift (Network Blindness).** Dummy columns in the RMP caused massive dual values ($100,000$), saturating the neural network's activation layers.
* *Solution:* Normalized dual values to $[0,1]$ for the NN input features, while keeping exact raw values for the OR reduced cost filter. Lowered dummy Big-M costs.


* **Problem 3: The Euclidean Fallacy.** POMO was trained on synthetic 2D coordinates, blinding it to the asymmetric, real-world OSRM distance matrix.
* *Solution:* Rewrote `InstanceGenerator` to sample subgraphs directly from real `orders.csv` data and query the actual distance/time matrices. Standardized exact cost formula: `Fixed + (Dist * CostPerKm) + (Time * CostPerHour)`.



### **Phase VI: Hardware Migration, Fallbacks, & Current Mandate**

* **Computational Shift:** Heavy RL training has successfully migrated to a **powerful remote workstation**, removing previous hardware bottlenecks for scaling. However, **design-phase testing and local validation remain strictly constrained to the 8GB RAM CPU environment**.
* **Active Engineering Mandate:**
1. **Dynamic Programming Fallback:** Building an exact ESPPRC Label Setting algorithm. Due to the 8GB testing limit, this uses an aggressive beam width to prevent local OOM crashes during design.
2. **3-Strike Logic:** Orchestrator will try POMO 3 times. If no negative RC columns are found, it defaults to the DP fallback.
3. **Genetic Algorithm (GA) Generator:** Building a GA to generate diverse (optimal and sub-optimal) candidate columns to feed the bipartite graph for DQN training.
4. **Extensive Benchmarking:** Training the new "real-world" POMO on `temp_days` data and running a 3-way benchmark: Pure DP vs. Pure POMO vs. POMO+DQN.