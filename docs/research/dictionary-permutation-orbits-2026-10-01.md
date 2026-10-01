# Separate emission inventory from source-letter assignment

2026-10-01. Own mathematical continuation during the fixed CONTEXTUAL-GIBBS-001 campaign. The registered sampler and all195frozen paths remain unchanged. No swap sampler or new corpus experiment is implemented or queued.

## Source review and scope

Key swaps are established cipher-search tools, not a new invention. [Dhavare, Low and Stamp, Efficient Cryptanalysis of Homophonic Substitution Ciphers](https://www.cs.sjsu.edu/~stamp/RUA/homophonic.pdf) discusses letter swaps and a hierarchy separating assignment search from homophone-count search. Selected PDFpages4,10–13and21–25 were read; some other swap-containing extracts were inspected, not a full-paper audit. Its digram scoring, deterministic inverse homophonic substitution and optimization objective differ from our variable-emission marginal-likelihood sampler. Its matrix shortcut and performance are not transferred. Prior Voynich application/semantic limitations remain in the [contextual review](native-censored-context-2026-10-01.md).

The browser PDF reader returned InternalError; the public author-hosted PDF was retrieved once with a10MiB/30sdownload cap, saved outside Git, and extracted with bundled pypdf.1971703bytes/SHA2562c2b4f7ada88d3d6cbbed47392dd7da217056d310f531684f9e64f68af69aa92. [Retrieval manifest](../../data/manifests/research_row_permutation_20261001.json). No third-party full text is added to Git.

## Own support argument

Assume the actual source probabilities are strictly positive at EVERYcontext/letter, transitions are valid, rho lies in(0,1), and emissions are nonempty. These are the current qualified native model's conditions. For a fixed dictionary K, a closed observation has positive likelihood exactly when it is segmentable using codes in image(K). The analogous unclosed prefix can end inside the final code. Every valid finite segmentation can be assigned letters emitting its units, and the corresponding finite source path has positive mass. Conversely any positive source path supplies such a segmentation.

A permutation of free source rows preserves image(K), every code multiplicity, all known row bindings and both observations' support. It can change source-language probabilities strongly. Thus swapping two rows changes linguistic labeling without deleting the code inventory that makes encoding possible. This prevents structural zero-target proposals under these assumptions; it does not bound graph size or prevent very unfavorable probability ratios.

Uniformly choosing a pair of free rows and swapping their codes gives a symmetric proposal. Exact MH acceptance is min(1,G_t(swapped)/G_t(old)), with oldpositive. Duplicate codes can yield self proposals; the involution still proves symmetry. Any fixed state-independent mixture of such swaps and an already-qualified invariant kernel remains invariant. All row permutations share an inventory orbit, so swaps alone cannot discover a missing code or alter its multiplicity.

The [finite checker](../../scripts/check_row_permutation_theory001.py) and [receipt](../../results/ROW-PERMUTATION-THEORY-001/result.json) verify6696exact-rational source-string support comparisons against independent Boolean code-inventory word break, and6696exact MH balance checks: all36two-row/six-code dictionaries,3contexts, every binary observation length1..3, all prefixes including0, and closure. The previous two-supported-key Gibbs barrier is crossed by a swap with forwardacceptance1/12/reverse1, without pretending probability1. A zero-probability-source witness gives original3/16/swapped0, so the positivity restriction is essential. This is finite arithmetic, not corpus scoring, empirical mixing or recovery evidence.

## Own factorization and what it can expose

For Rsource rows and Ucodes, a code-count vector m has sumR. Under independent uniform code choices,

P(m) = R! / (product_u m_u! × U^R).

Given m, distinct source-row assignments are uniform over R!/product_u m_u! possibilities. Swaps explore those assignments; changes in counts explore inventory. The finite checker verifies all216three-row/six-code dictionaries partition into56histograms with these exact orbit sizes. ForR23/U42, count vectors number C(64,23)=146721427591999680, while full dictionaries number42^23=21613926941579800829422581272845221888. Neither space is tractable by exhaustive search. This factorization separates two problems; it does not automatically reduce total inference complexity.

Let c_u be the number of ACTUALLYUSEDgold source rows requiring code u, and m_u the current dictionary's code counts. The maximum correct used-row assignments attainable by ANYrow permutation is sum_u min(c_u,m_u). The deficit sum_u max(0,c_u−m_u) is the minimum number of code replacements needed before some permutation can match all used rows, ignoring intermediate likelihood/support constraints. This is an oracle diagnostic only: c_u is unknown in Voynich. Current labels, inventory ceilings and likelihood preferences must be kept separate.

Holding inventory fixed is therefore a useful future causal control: measure changes in contextual scores and recovered labels while code coverage/multiplicities remain constant. Contextual scores generally change, and even nonuniform IID scores can change because source-letter frequencies differ; histogram preservation is not IID-likelihood invariance for the actual nonuniform source. Swaps may help coordinated relabeling, but they cannot fix an incorrect inventory or an inadequate source model. A bounded future sampler needs its own kernel/zero-source/known-binding/resource tests and registration. Historical decipherment remains unresolved.

## A stronger inventory barrier, checked separately

For two closed observations01and10and two source rows, exhaustive enumeration of all36legal dictionaries finds exactly four supported keys: (0,1),(1,0),(01,10),(10,01). Single-row replacements cannot leave any of those keys. Adding row swaps connects each pair but still leaves two separate inventory components. A full-pair uniform proposal connects all four with positive exact MH probabilities.

[Separate finite checker](../../scripts/check_inventory_barrier001.py) and [receipt](../../results/INVENTORY-BARRIER-THEORY-001/result.json) verify all36likelihood supports, three exact4×4stochastic transition matrices and48detailed-balance pairs. Thus swaps address a genuine labeling barrier but cannot generally address inventory barriers. This is an exact counterexample, not proof that these components cause the large23-row campaign's failures. Prefix targets, spare source rows and alternative segmentations can change connectivity before closure; global or inventory-block updates need their own qualified construction.
