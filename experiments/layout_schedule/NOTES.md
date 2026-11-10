
## Deep-probe findings (synth recall, one-hop)
ctx_sensitivity (relative change when earlier tokens scrambled) per layer:
- **B (attention-ends)**: peak L1-L2 (1.31-1.33), high through L4-L11 (~0.9), drops L12+ (MLP). -> context read EARLY. EM 1.0 (best).
- **A (attention-middle)**: L0-L1=0 (MLP, no ctx), rises to peak L8-L13 (~1.08). -> context read LATE/middle. EM 0.41 (worst).
- **baseline (alternating)**: moderate early, high L4-L12, drops L13+. EM 0.95.
CONCLUSION: recall needs attention at the ENDS (early read + late readout). Middle-only attention (A) fails.
This directly supports the hypothesis: placement of A vs M matters; it's not interchangeable.

## Synth task summary
- local_rule (token-local): SATURATED, all layouts equal (control). EM 1.0.
- recall (one-hop): STRONG layout effect. B > baseline >> A. Attention-at-ends wins.
- hop2 (multi-hop): nearly saturated, minimal effect. All ~0.99 EM.

## RECALL full ranking (16 runs, 16 sublayers, early|mid|late A-counts)
| variant | layout | A-counts | EM    | loss  |
|---|---|---|---|---|
| D | AAAAM MMMMM MMAAA | 4-0-3 | 1.000 | 2.098 |
| B | AAAMM AMMAM MAMMA | 3-2-2 | 0.998 | 2.088 |
| baseline | AMAM... | 3-2-3 | 0.950 | 2.097 |
| sandwich | AAAAM AMAMA MAMMM | 4-3-1 | 0.905 | 2.098 |
| C | MMMMA AAAAA AAMMM | 1-5-2 | 0.680 | 2.177 |
| A | MMAAM AAMAA MAAMM | 2-4-2 | 0.413 | 2.224 |
MECHANISM: early attention binds key->value (reads context). Strong early attention (B,D) -> best.
Concentrating attention in the middle (A,C) with weak early attention -> fails.
Matches ctx_sensitivity: B/D read context early (high L0-L2), A/C don't.
=> The "best" layout is TASK-SPECIFIC (recall wants early attention; hop2/local_rule saturated).
