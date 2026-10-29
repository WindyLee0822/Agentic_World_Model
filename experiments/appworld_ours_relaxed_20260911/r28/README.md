# R28: Direct visibility of the static write contract

R27 diagnostics captured a failed payment-write attempt whose generated programs guessed response fields, fabricated an ID and exhausted repair on disallowed operations. The current contract was available in Python context, but the generated comments still guessed its contents.

R28 additionally renders the static current endpoint documentation and success/failure schemas directly in the opt-in write prediction prompt. State evidence, request arguments and write history remain in their existing structured context. The simulator still decides the outcome and generates the response; no schema example becomes a host-generated response. Verifier and selection behavior are inherited from R27.

Only the opt-in subclass changes versus R27; all other Python source files are identical. A two-task N=1 diagnostic configuration is prepared (normal/L1 indices 9,34). No performance claim is established, and R22's full BoN@16 run continues independently.
