# JOINT-KEY-TRAIN-001 — prepared and replayed inputs

2026-09-30. Source/protocol18cb24ba8f7e5a37e5472a9ff6251d02e45faf10was pushed and
the exact remote ref verified before one input preparation and one input audit.
Both completedPASS with no redraw/retry/extension, no model fitting, no cipher-panel
input, no reserved reader author and no paid API/cloud use.

| Source role | Eligible letters | Retained segments | Policy |
| --- | ---: | ---: | --- |
| Training | 6,497,939 | 2,083 | Same12allowed authors, segments>=512letters |
| Validation | 351,029 | 358 | Pliny selection author, segments>=224letters |

Validation's original359,276letters lose8,247to the declared segment filter.
Training excludes319,711letters consistently for both model sizes. Original
source IDs, author roles, boundaries, text checksums and corpus manifest remain
bound in`results/JOINT-KEY-TRAIN-001/inputs.json`. The existing64letter cross-role
overlap exclusions and corpus provenance/rights caveats are inherited.

All64fixed validation episodes have their source-window coordinates, literal
23-row keys, canonical keys, observed/unseen-symbol inventory, diagnostic used
rows and exact ciphertext hash recorded. Literal/canonical full-key uniqueness
holds across all64. Preparation uses only the declared seed72221, no empirical
outcome or favorable redraw. Both finite key sets will be excluded from every
training draw. Validation is reused selection material, not new qualification.

One input audit reopens the allowed original source roles, compares every stored
training/validation text and source profile, reproduces all64episode metadata
from the fixed seed, and checks literal/canonical/window identities. It is a
same-author algorithmic replay, not independent-agent approval.

Preparation .326347wall/.315144CPU seconds,268,435,456peakRSSbytes; input audit
.100487wall/.095643CPU seconds,272,596,992peakRSSbytes. These stage times exclude
imports/admission. Both original600wall/500CPU/4GiBcaps preserved. Training and
validation texts and episode metadata are compressed outsideGit; compact
checksummed manifests/receipts only are published.

Full2,447tests+23subtests passed/13skips before preparation,11focused including a
real tiny four-fit optimizer/archive/complete-audit integration and record
duplication gradient invariance. New code Ruff/diff passed. Model/source/protocol
bytes remain identical to18cb24b. No actual language-training update or recovery
measurement occurred in preparation. Publish these audited inputs, verify the
remote ref, then start the single registered bounded MPS campaign.
