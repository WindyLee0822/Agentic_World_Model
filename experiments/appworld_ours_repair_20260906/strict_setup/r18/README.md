# R18 strict observation evaluation

Job: `dw-appworld-ours-strict-r18-full-20260907`.

Same all-585-task N=2 evaluation and simulator selection as R17. The sole runtime change suppresses native setup `show=True` stdout, then prints only the generated prediction. This closes a reachable observation leak in R17. Native task writes remain blocked during candidate simulation.

25 focused routing, generated Python and strict result-gate tests passed before submission. R17 saved candidate action signatures truncate each action at 400 characters, so their lack of `show=` cannot prove observation compliance. R17 is superseded; do not combine its results with R18.

Source stage: `/tmp/appworld-strict-r18-stage`. Durable changed module and routing tests are in `../show_hardening/`; all other runtime files match submitted R17. Root runtime has not yet been promoted.

Full performance validation is pending. Poll with `poll.py` in this directory; `live-status.json` is authoritative only as of its recorded check time.
