# What the larger investigation actually found

**We found a concrete weakness in our intervention method: where we edited the model mattered enormously.** The first completed program used an existing eight-billion-parameter model, about3hours42minutes of registered local computation, and12,168 intervention trials. A later fresh two-slot attention-path study added2,048 discovery head interventions and192 confirmation generations in16m34s; it also finished. No model job from these studies remains active.

The first tests were disappointing. Changing our selected internal country coordinates redirected only17/151 eligible final answers. Changing64 selected neurons using information from a different question redirected0/31. Both failed their preset criteria, and those failures are preserved.

We then tested a specific explanation. Our edits had touched only the final position of the prompt. The model could still access information stored at earlier positions. On a small development subset where both original questions were answered correctly:

| What we changed at block24 | Answers redirected as intended |
| --- | ---: |
| Complete donor state at the final position | 0/16 |
| Complete donor states at earlier positions | 15/16 |
| Complete donor states across the prompt | 16/16 |
| Narrow country-coordinate edit across the prompt | 5/16 |
| Random edits of matched size across the prompt | 0/16 |

Copying unrelated words survived12/12 in every condition. At block28, the final-position donor edit became much more effective (12/16), while earlier-position replacement became less effective (4/16). This suggests that useful information is routed toward the answer position as the model processes the question.

**The implication:** an internal representation can be readable without being a reliable control switch. We need to investigate how information moves between positions and components, and whether other routes restore the original answer after an edit. Looking only for a few highly active neurons misses that possibility.

The16-case follow-up is exploratory and heavily dependent: it reuses development facts, includes reverse directions and paraphrases, and covers only two eligible country pairs. The model's imperfect codebook competence leaves many cases outside that denominator. Replacing a complete state also changes far more than one concept. The narrower coordinate edit remains unreliable, and no Voynich meaning, encoding rule or translation has been established.

A fresh binding follow-up then asked whether one attention block carries the information. Its clean model answered every explicit two-slot binding in both discovery and confirmation sets. Scanning heads on discovery selected four in block25; on disjoint confirmation words they redirected5/16 answers, while a donor-value patch at changed record-token positions redirected3/16 and a matched random head patch0/16. But even replacing all32 heads at that block redirected only3/16. The registered positive control therefore failed, making the single-block path claim **uninformative**. The result shows some task-specific causal leverage but also how misleading it would be to call the selected heads a complete circuit. See [PATH-0001 results](../../experiments/PATH-0001-results.md).

Validation:740 repository tests passed, four optional checks skipped,23subtests passed;19 separate live architecture checks passed. All scored output labels, identity controls, source hashes and compact archives were checked. Derivative arrays and387 saved baseline/trace arrays passed checksum verification. Scientific figures were visually inspected.

Detailed reports: [Jacobian-lens study](../../experiments/JSPACE-0001-results.md), [neuron study](../../experiments/NEURON-0001-results.md), [position-coverage diagnosis](../../experiments/ROUTE-0001-results.md). The next justified question is which attention paths perform this transfer, with confirmation on genuinely new inputs—not another reuse of the consumed final set.
