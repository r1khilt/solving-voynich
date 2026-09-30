# The two catastrophic fresh failures are missed fitting solutions

2026-09-30. All16 exposed positive cases complete under the separately frozen
[diagnostic registration](BLIND-CHANNEL-CONFIRM-002-DIAG-A.md). No key was fitted,
no transfer passage re-decoded, and the original fresh recovery/screen **FAIL**
remains. This diagnosis identifies known better fitting candidates, not global
optimality or a new successful decipherment.

The correct dictionary beats the selected dictionary under **all three**
fitting objectives for both catastrophic cases. The largest source makes the
advantage especially large. Each table entry is learned cost minus generating
cost in bits; positive means the generating dictionary is better. Costs include
the unchanged literal dictionary description.

| Fresh key /opaque case | Order1 gap | Order3 gap | Large-source gap | Best saved endpoint versus expanded selection | Most matching rows among16endpoints |
| --- | ---: | ---: | ---: | --- | ---: |
| 7 /case32 | +329.874305 | +681.189916 | +1772.409524 | 74.882278bits better; restart00 | 3/23 |
| 13 /case01 | +133.977644 | +403.087303 | +1331.795309 | 86.770252bits worse; restart13 | 7/23 |

All16 saved restarts for each of these two cases stopped at local optima.
They are certificates for the tested replace/swap neighborhoods under order1,
not global optimality. The correct dictionary has a better score even under
that original objective. Therefore the bad result cannot be explained solely
by an objective that ranks that selected dictionary above the truth.

Selecting a different saved endpoint alone is insufficient evidence of a cure.
Case32 has one endpoint with a better large-source fitting cost, but its mapping
matches0/23true rows. Its fitting improvement does not establish better reading.
Case01 has no saved endpoint beating the expanded selection. Even its largest
row match is only7/23. Merely rescoring the same16final dictionaries cannot
recover a good dictionary that they never found. Refining more basins remains
an untested method, not an observed recovery.

Across all16 cases, the generating dictionary is a known better candidate than
the corresponding selected dictionary for4order1 cases (01/10/25/32),
3order3 cases (01/10/32), and2large cases (01/32). For the other14large cases,
the selected dictionary has a lower fitting cost than the generating dictionary
by0.052158–15.829916bits. This does not prove a wrong key is identifiable or
globally optimal; unseen assignments, description costs and source probabilities
remain relevant. It does reinforce the distinction between the two catastrophic
search failures and the smaller covered-case source/decision errors.

## Complete checks and resource record

Source/protocol and original failed evaluation were published and exact remote
verified at2812e2c7c33051cb49bd741f10564f48362fba2e before the single run
(session39603, terminalexit0). All256restart endpoints and320case/label results
remain in the record. Equal dictionaries share computation, yielding1,208native
large-source record scores rather than the1,280maximum allocation. All64large
generating-key record scores agree with independent string-context backward
inference; all128original order1/order3 record scores agree with separate
manual backward implementations. Overall maximumdelta1.0232e-12nats, below
the frozen1e-7 tolerance. Unchanged parent/best scores match their published banks.

Measured22.012826wall/21.924231CPU seconds,3,079,766,016bytes peak host RSS,
one CPU process/one numerical thread,0paid. All source, input, answer, trace and
result bindings pass the alternate arithmetic/inventory audit; it checks all320
score labels, model-code costs, sums, comparison directions, selected endpoints,
literal row counts and the1,208work count. Exact audit code bytes/SHA are saved.
It computes no new likelihoods or predictions. Root authored alternate checks
are not independent-agent review. No limits were increased or run retried.

Prelaunch full2,268tests+23subtestsPASS/13skips; targeted10PASS, changed
source/tests Ruff and diff hygiene clean. Full-tree Ruff retains exactly the
five earlier unrelated findings. Bulk sources/traces remain ignored.

## Consequence

The next primary target is global dictionary search that can cross between
bad local basins, using the already strong exact source. Multiple retained
basins, nonlocal proposals and bounded partial-dictionary search need controls
and new experiments. The data do not justify assuming that more local rounds,
another source model, or neural interpretability alone will repair these cases.
Even finding a better-scoring dictionary does not guarantee the true global
optimum or semantic correctness. Fresh qualification remains necessary after
development changes, and a bridge to a historical cipher is still unresolved.

Machine records: [result](../../results/BLIND-CHANNEL-CONFIRM-002-DIAG-A/result.json),
[all-case accounting](../../results/BLIND-CHANNEL-CONFIRM-002-DIAG-A/accounting.json).
