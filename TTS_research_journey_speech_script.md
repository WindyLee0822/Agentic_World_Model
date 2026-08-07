# Speech Script: Agentic World Modeling for Agent TTS

**Suggested duration:** 12-15 minutes  
**Source convention:** The research motivation and GES formulation follow *Agentic World Modeling for Agent Test-Time Scaling without Environment Replica*. Concrete system behavior refers to this repository.

## Slide 1 — Title

Good morning. Today I will present our research journey toward agentic world modeling for test-time scaling.

The problem is simple to state but difficult to solve. Test-time scaling works by sampling multiple candidate trajectories and choosing a good one. For an agent, however, a trajectory contains real tool calls. We cannot safely execute many speculative trajectories against the same enterprise workspace, and in many realistic environments we also cannot create a separate writable replica for every rollout.

Our central idea is to replace speculative execution with grounded next-state prediction. The world model predicts what each tool call would return, but it can interact with the real environment through bounded, read-only queries before making that prediction. This lets us spend more inference-time compute without allowing candidate trajectories to interfere with the real state.

## Slide 2 — Reformulation: Context Becomes an Interaction

The key conceptual shift is that context becomes an interaction.

A conventional simulator receives one compressed prompt and immediately predicts the next observation. Our simulator instead follows four steps. It observes the candidate action, identifies which state is uncertain, issues bounded read-only probes, and then computes the observation.

This is the basic GES formulation from the paper: next-state prediction is no longer a single-shot generation problem. It becomes an evidence-seeking agent task. The evidence is action-conditioned, because the simulator waits until it knows the hypothetical action before deciding what information is relevant.

This distinction is important. We are not trying to place the complete environment into the prompt. We are allowing the simulator to ask focused questions about the one transition it currently needs to predict.

## Slide 3 — Why a Compressed Prompt Is Not Enough

Why not simply build a larger prompt?

A static prompt must anticipate every state variable, dependency, policy rule, and failure mode before it knows which action the candidate agent will take. If we include everything, the context becomes long and expensive. If we summarize or retrieve only part of the state, we risk excluding the exact fact that becomes important later.

The paper formalizes this as an information-allocation problem. An action-agnostic summary must divide a fixed context budget across many possible future actions. Action-conditioned retrieval can spend that same budget on the fact required by the action that was actually proposed.

So our solution is not merely a better summary. It is a sequential information-gathering process with an explicit stopping condition.

## Slide 4 — Compressed Simulation versus Agentic Simulation

This example makes the difference concrete.

Suppose a candidate trajectory has already updated a range of customer records and now asks for the final coupon for record 123. A compressed simulator may remember the promotion rule but omit the customer's region or VIP status. It then produces a plausible but incorrect answer, such as 15 percent.

The agentic simulator can first read customer record 123, then inspect the relevant promotion rules, and finally use a restricted computation step to combine the evidence. In this example, the base rule, VIP bonus, and regional cap produce the grounded answer of 12 percent.

The point is not that more reads are always better. The point is that the simulator chooses reads based on the transition. It stops when it has enough evidence or when it reaches its read budget.

## Slide 5 — The Next Engineering Problems

This reformulation introduces two engineering problems.

First, each trajectory contains many predicted transitions, and each transition may contain several evidence-gathering turns. If all of those turns remain in one conversation, the context grows very quickly.

Second, a flexible read-only tool set creates a large safety and parsing surface. Free-form textual commands are difficult to validate consistently, especially when the model needs several reads and an exact structured result.

Our implementation therefore introduces two complementary mechanisms: isolated prediction and verification subagents for context management, and a restricted Python protocol as the structured control plane for evidence gathering and computation.

## Slide 6 — Agent and World Model Interaction

This diagram shows the latest architecture.

On the left is the official benchmark agent. It chooses tool calls, receives observations, and continues the normal task dialogue. It does not receive privileged database state.

The router in the middle is ordinary harness code, not another LLM. It intercepts each tool call, tracks the action history and whether a read is clean or depends on a prior predicted write, and routes the request to the correct component.

Inside the world model, the observation predictor runs for each routed tool call. It checks preconditions, gathers read-only evidence when needed, predicts the exact observation, and updates candidate-local logical state. After the trajectory ends, a separate success verifier audits the actions and predicted final state against native or reference-grounded criteria.

Both subagents can use their own bounded read-grounding loops, but the host executes and validates every real read. Neither subagent receives a database handle or a write-capable environment object.

In the repository, this separation is implemented by the environment-specific predictors and verifiers. For OfficeBench, the predictor is in `src/tts/env/officebench_wm_predict.py`, while terminal success verification is in `src/tts/env/officebench_wm.py`. The generic read-grounded simulator, budget, and spawn-tree accounting are in `src/tts/worldmodel.py`, `src/tts/budget.py`, and `src/tts/simulator/spawn_tree.py`.

## Slide 7 — From Textual Tools to Python-Only Mode

Our first implementation exposed textual commands such as READ, LIST, CALL, and TRACE directly to the model.

Those capabilities were useful, but the interface was command-oriented and usually handled one operation per turn. Parsing free-form text also made exact output and safety enforcement harder.

The current design keeps the same read-only capabilities but moves them behind one structured protocol. The model emits restricted Python that either requests evidence or computes the final result. This gives us one control plane for batching, validation, exact transformation, and failure handling.

## Slide 8 — The Capability Did Not Disappear

This change is an interface migration, not a reduction in capability.

Previously, the model emitted READ, LIST, CALL, or TRACE as top-level textual commands and finished with PREDICT or VERDICT. Now it emits `PYTHON_READ` to construct a validated JSON query array. The host executes only approved read-only operations and copies the results into variables such as `data['reads']` and `data['facts']`. The model then emits `PYTHON_FINAL` to compute the exact observation or verdict.

The important safety property remains unchanged: tool access is host-validated and read-only. The generated Python does not directly execute an OfficeBench action or access the task environment.

The implementation is in `src/tts/env/officebench_wm_python.py`, with behavioral coverage in `tests/test_officebench.py`.

## Slide 9 — Python Protocol Example

Here is one complete interaction.

First, a `PYTHON_READ` program prints a JSON `queries` array. Second, the host parses the schema and rejects unsupported or mutating requests. Third, the approved READ, LIST, CALL, or TRACE operation is executed through the read-only query interface. Fourth, the result is copied into the restricted context. Finally, a `PYTHON_FINAL` program transforms those copied values and prints the structured observation or verdict.

The restricted program receives a `data` dictionary, but it receives no session object, filesystem handle, application object, network access, subprocess access, or process environment.

The loop is explicitly bounded. In `run_python_prediction_loop`, the repository limits the number of turns and reads per turn. If the read budget is exhausted, the model must return a final program. Invalid programs can receive bounded repair attempts; required Python mode fails closed rather than silently falling back to an ungrounded direct answer.

## Slide 10 — Test-Time Scaling

With a safe simulator, we can return to test-time scaling.

For each task, we sample N candidate trajectories using the same official agent prompt and sampling path. We simulate each trajectory without mutating the real environment, estimate its probability of success, select one candidate, and then officially evaluate only that selected trajectory.

The repository makes the safety contract explicit: zero simulator writes, one real commit, and N candidate trajectories.

`src/tts/harness.py` orchestrates this process. `src/tts/selection.py` ranks candidates using signals such as simulator success or graded simulator confidence. `src/tts/commit.py` implements a single-use `CommitChokepoint`; a second commit raises an error. The harness also compares the environment hash before and after sampling and records `simulator_writes`, so the no-write invariant is auditable rather than only assumed.

## Slide 11 — Experimental Setting

We evaluate the approach across three different interaction modalities.

Tau-bench represents tool-agent-user interaction in customer-service domains. AppWorld uses a stateful Python agent across multiple applications. OfficeBench uses office-application actions and execution-based evaluation over files and application state.

The comparisons include rule-based or flat simulation baselines and a trained world-model baseline. We test both non-reasoning and reasoning frontier models, as well as an open-weight Qwen world model.

The important fairness principle in the implementation is that candidate generation is method-independent. The same official prompts, seeds, and rollout pool are used across methods where possible. The method changes the simulation and selection signal, not the benchmark agent's task prompt.

## Slide 12 — Experimental Results

The current results show two main trends.

First, grounded simulation improves the selection signal over compressed-context simulation, especially as N grows. In the displayed GPT-5.4 results, the gain is large on difficult multi-step settings. For example, on AppWorld Level 2 at best-of-16, GES reaches 93.94 compared with 58.59 for the compressed SIMIA baseline. On AppWorld Level 3, the corresponding values are 93.02 and 57.75. Tau-bench Retail also increases from 75.44 to 91.23.

Second, the effect varies by benchmark and difficulty. OfficeBench gains are more moderate in some columns, which tells us that better transition prediction is necessary but not sufficient; native evaluation fidelity and long-horizon state tracking remain important.

These results currently focus on simulation beginning at the first write action. Simulation from the beginning, including clean reads under predict-all mode, is the stricter setting and is the direction represented by the latest implementation.

## Slide 13 — Fine-Tuning and Tool-Protocol Comparisons

This slide highlights two additional observations.

First, a strong general model with the GES scaffold can outperform a world model trained specifically for simulation. On AppWorld Level 1, the GPT-5.4 row remains above the Qwen-WM row across the shown compute settings. This supports the paper's claim that interactive grounding can complement, and sometimes substitute for, specialized world-model training.

Second, the tool and computation interface matters substantially. Removing the Python tool causes a large drop on AppWorld Levels 2 and 3. At Level 2 and best-of-16, the displayed score falls from 74.24 to 61.12. At Level 3, it falls from 63.95 to 50.38.

The reason is not simply that Python adds more intelligence. It provides reliable batching, filtering, exact copying, and arithmetic over host-grounded evidence. It also creates a clearer safety boundary than free-form commands.

## Slide 14 — Next Experiments and Closing

The next step is to treat simulation quality itself as a new axis of test-time scaling.

We can scale the number of candidate trajectories, but we can also scale the computation used inside each transition prediction: more evidence-gathering rollouts, stronger aggregation, or more careful verification. These axes may be complementary.

We also need broader comparisons with aggregation methods, more ablations on read budgets and stopping rules, and a clearer decomposition of errors from candidate generation, observation prediction, final-state verification, and selection.

The broader message is that agent test-time scaling does not require us to fork the world. By turning next-state prediction into a bounded, read-grounded interaction, we can explore multiple hypothetical trajectories while preserving one real environment, zero speculative writes, and one auditable final commit.

Thank you.

