# Explicitly synthetic interface fixtures

These small JSON files demonstrate the evidence, grounding and transition-program interfaces. They contain invented observations and likelihoods, not manuscript evidence. They are never included in model training or scientific evaluation.

From the repository root:

```sh
.venv/bin/python -m voynich.communication plan-evidence --belief examples/communication/belief.json --queries examples/communication/queries.json --output outputs/communication-fixtures/queries.json
.venv/bin/python -m voynich.communication update-belief --belief examples/communication/belief.json --evidence examples/communication/evidence.json --output outputs/communication-fixtures/updated-belief.json
.venv/bin/python -m voynich.communication ground --candidate examples/communication/candidate-graph.json --observed examples/communication/observed-graph.json --text-source-group fixture-text --output outputs/communication-fixtures/grounding.json
.venv/bin/python -m voynich.communication extract-transitions --examples examples/communication/transitions.json --validation examples/communication/transition-validation.json --output outputs/communication-fixtures/transitions.json
```

Expected behavior: the discriminating query has information gain 0.3680642072 nats and the uninformative query has zero; the evidence update gives probabilities 0.9/0.1; graph matching preserves two equally good entity alignments; transition evaluation reports one known and one unknown transition. Perfect edge agreement does not remove alignment ambiguity. The reused evidence source must be rejected if applied a second time.
