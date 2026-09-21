# AI on hard problems, and what transfers to Voynich

**Research memo — 2026-09-21.** Literature and method review only. No new training, no paid research API, no decipherment claim. This document does not modify `docs/RESEARCH_CHARTER.md`. It is complementary to the 2026-09-21 deep review and world-model/diffusion notes; it does not replace [NEXT_DESIGN.md](../deep-review-2026-09-21/NEXT_DESIGN.md) or [DECIPHERMENT.md](../deep-review-2026-09-21/DECIPHERMENT.md).

**Claim labels used here**

- **Sourced observation:** a named primary source, fetched for this memo, with URL in [SOURCES.md](SOURCES.md).
- **Working hypothesis:** a testable mapping from those observations onto this repository’s protocol.
- **Speculation:** explicitly labeled; not evidence.

Every “AI solved X” headline below is classified as one of: numerical approximation / surrogate; scientific discovery of an object or bound with an independent check; formal proof search; conjecture/counterexample search; or an actual resolution of a named open problem (with remaining institutional caveats).

---

## 1. Sourced findings

### 1.1 Navier–Stokes: four different things that get conflated

**Official problem (sourced).** Clay’s official description (Fefferman) asks for a proof of **one** of four statements about 3D incompressible Navier–Stokes with viscosity \(\nu>0\):

- **(A)** global smooth finite-energy solutions on \(\mathbb{R}^3\) for every admissible smooth divergence-free initial field, **with force identically zero**.
- **(B)** the same on the torus \(\mathbb{R}^3/\mathbb{Z}^3\), force zero.
- **(C)** existence of some smooth initial field **and** some smooth force on \(\mathbb{R}^3\) for which no global smooth finite-energy solution exists.
- **(D)** the analogous breakdown statement on the torus.

Source: Clay PDF [S01](https://www.claymath.org/wp-content/uploads/2022/06/navierstokes.pdf), fetched 2026-09-21. Two-dimensional analogues of (A)/(B) were already known (Ladyzhenskaya, as cited there). Small-data and short-time 3D regularity are also classical; the prize question is the remaining alternative.

**Neural PDE solvers did not resolve that question (sourced).** They are **numerical approximation / operator learning**:

- **PINNs** (Raissi, Perdikaris, Karniadakis, *J. Comput. Phys.* 2019) train a network so that PDE residuals, initial/boundary data, and (optionally) observations enter a loss. The authors demonstrate data-driven **solution** and **discovery of PDE coefficients** on classical fluids/quantum/reaction–diffusion examples [S10]. This is not a existence/smoothness theorem.
- **Failure of the same idea as a general solver (sourced).** Krishnapriyan et al. (NeurIPS 2021) show vanilla PINNs can fail on convection/reaction/diffusion problems because the PDE residual as a *soft* constraint yields an ill-conditioned landscape, not because the network lacks expressivity [S11]. Curriculum regularization and sequential time-stepping reduce error in their tests. **Implication:** a physics-looking loss is not a verifier.
- **DeepONet** (Lu, Jin, Pang, Zhang, Karniadakis, *Nat. Mach. Intell.* 2021) learns operators (branch+trunk) from data; applications include ODE/PDE maps [S12]. **Fourier neural operators** (Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart, Anandkumar, ICLR 2021) learn parametric PDE maps in Fourier space; reported Navier–Stokes experiments are turbulent-flow **surrogates**, including zero-shot super-resolution versus prior ML solvers, with inference much faster than traditional solvers [S13]. These papers do not claim a Clay statement.
- **SINDy** (Brunton, Proctor, Kutz, *PNAS* 2016) recovers sparse ODEs from derivative data when the true dynamics are sparse in a chosen library [S14]. **AI Feynman** (Udrescu & Tegmark, *Sci. Adv.* 2020) recovers 100 Feynman-lecture formulas and 90% of a harder physics test set by combining neural fits with physics-inspired reductions [S15]. Both are **equation discovery under a library/symmetry prior**, not millennium proofs.

**2026 OpenAI claim: forced finite-time blowup, not unforced global regularity (sourced, with caveats).**

- OpenAI’s technical PDF, fetched 2026-09-21 from `cdn.openai.com`, states Theorem 1.1: for every \(\nu>0\) there exist a smooth compactly supported force, compact spatial support, **zero initial velocity**, and smooth \(u,p\) on \(\mathbb{R}^3\times[0,1)\) such that energy stays bounded while \(\|u\|_{L^\infty}\) blows up as \(t\uparrow 1\) [S02]. That is a **breakdown with a designed force**, i.e. in the direction of Clay **(C)** (and the repository metadata quoted in search also claims **(D)** on the torus [S04]). It is **not** a proof of (A) or (B) (unforced global smoothness for *all* data).
- Nature news (Castelvecchi, 8 September 2026) reports OpenAI’s 8 September announcement, quotes Ven Chandrasekaran that solutions can achieve infinite speed in finite time, and quotes CMI president Martin Bridson as calling it “an exciting day” while contemplating “major advances,” not as awarding the prize [S03]. The same article reports competing 7 September Euler (inviscid) claims by Alpöge & Buckmaster (using Claude/Codex/Astra) and by Anandkumar and collaborators using a PINN; **those papers were not fetched here**.
- Clay’s 11 September 2026 announcement says the Navier–Stokes problem has “**apparently been settled**,” that prize evaluation is “deliberately unhurried,” and that updates will follow [S05]. **Sourced observation:** CMI has not, in that text, awarded the $1M prize.
- CMI prize rules (2018 revision, official page): CMI does **not** accept direct submissions; consideration requires publication in a qualifying outlet, **at least two years** after that publication, and **general acceptance** in the mathematics community [S06].
- OpenAI’s HTML blog returned HTTP 403 to our fetcher. Lean formalization is advertised; Lean Reservoir metadata (via search snippet, not a live `lake build` in this repo) lists a public `openai/NavierStokesAndEuler` project targeting (C) and (D) plus an unforced Euler blowup [S04]. **This project did not compile that Lean development.** Kernel acceptance, if independently confirmed later, would still leave: (i) fidelity of Lean definitions to Fefferman’s PDE; (ii) journal review; (iii) the two-year/acceptance clock.

**Classification of the 2026 claim.** If the analytic argument and formalization survive community checking, this would be **formal/analytic resolution of Clay alternatives C and D** (forced breakdown), **not** of A/B, and **not** a demonstration that PINNs or FNOs solved the millennium problem. Until qualifying publication and expert consensus, it remains a **public, machine-assisted proof claim**.

**Working hypothesis (not a theorem):** the scientifically transferable piece is not “train a bigger PDE net.” It is **search + an independent checker** (here: analysis + Lean) on a problem whose statement is already fully formal.

### 1.2 Combinatorial discovery with a cheap exact verifier

**FunSearch (sourced).** Romera-Paredes et al., *Nature* 2023: an LLM proposes **programs**; an automated evaluator scores them; an evolutionary loop keeps high-scoring programs [S20][S21]. On the **cap set** problem (largest subset of \(\mathbb{Z}_3^n\) with no three terms in line): they report new constructions, including a **512-point cap in dimension 8**, improving a previous 496 construction (DeepMind code/notebook [S22]), and an improved **asymptotic lower bound**. This is **scientific discovery of constructions**, verifiable by a finite combinatorial check. It does **not** determine the exact cap-set size for all \(n\), and it does not “solve combinatorics.” The same paper reports improved **online bin-packing heuristics**, scored by packing cost—not a theorem.

**AlphaTensor (sourced).** Fawzi et al., *Nature* 2022: AlphaZero-style RL searches tensor decompositions (matrix-multiplication algorithms). Reported: a \(4\times 4\) algorithm over \(\mathbb{Z}_2\) with **47** multiplications vs Strassen two-level **49**; other size-specific improvements; hardware-specific speedups [S23][S24]. Classification: **algorithm discovery with an algebraic verifier** (the factorization multiplies correctly). It does **not** determine the matrix-multiplication exponent \(\omega\).

**AlphaEvolve (sourced).** Novikov et al. white paper, arXiv:2506.13131 (16 June 2025): evolutionary coding agent; on >50 math construction problems, they report matching prior SOTA on ~75% and beating it on ~20%, including a **slightly improved upper bound** on **Erdős’s minimum overlap problem** (\(C_5\le 0.380924\) vs prior record they cite) and an 11-dimensional kissing-number construction [S25]. Georgiev, Gómez-Serrano, Tao, Wagner, arXiv:2511.02864, and Tao’s 5 November 2025 blog: 67 problems; often rediscovery; some improvements; **explicit failures** also reported; AlphaEvolve is for **scorable constructions**, not arbitrary theorems; they pipeline constructions into Deep Think / AlphaProof in some cases [S26][S27]. Classification: **construction search with a numeric/combinatorial evaluator**, sometimes followed by human or formal proof of a pattern.

**Erdős problems more broadly (sourced, incomplete).**

- **Misadvertised “solves”:** secondary press (October 2025) reported OpenAI staff claiming GPT-5 had solved previously open Erdős problems, then retracting after Thomas Bloom and others clarified that the model had **found existing literature** for items listed as open on Bloom’s site because *he* had not yet recorded the papers [S40]. **We could not fetch the original X posts or Bloom’s contemporaneous comment in this pass** (erdosproblems.com returned a Cloudflare interstitial [S41]). Treat the episode as a **documented failure mode**: literature search ≠ new proof. Do not cite tweet counts we did not open.
- **Later, narrower, better-specified claims (sourced at second hand from the same site’s search snippets):** Problem **#846** (Erdős–Nešetřil–Rödl) is listed as **disproved** independently by a DeepMind Lean agent (21 February, year given as 2026 in the snippet) and an OpenAI internal model, with Rödl noting a short derivation from Reiher–Rödl–Sales 2024 after projection [S42]. Problem **#793** is listed **PROVED (LEAN)** with credit to GPT 5.6 Sol prompted by Chojecki [S43]. **We did not independently read those Lean files or the OpenAI PDF.** Classification if the Lean artifacts check: **formal refutation/proof of specific listed problems**, not a sweep of “the Erdős problems.”

**Grok / xAI (sourced vs not).**

- Viral “Grok 3 proved Riemann” was described in secondary coverage as a **joke retracted the same day** [S44]. **Not a result.**
- arXiv:2605.05193, “Grokability in five inequalities”: authors report **five inequality/bound improvements** found with Grok and **verified by the authors** (Gaussian perimeter, Hamming-cube moments, autoconvolution, \(g\)-Sidon sets, Szarek) [S45]. Classification: **human-verified bound tightening**, not a millennium problem.
- 2026 press about Grok 4.5 “hypercontractivity on \(S^4\)” / graph conjectures: **no primary paper fetched; not treated as fact.**

### 1.3 Formal math: IMO systems, not a general oracle

**AlphaGeometry (sourced).** Trinh, Wu, Le, He, Luong, *Nature* 2024: neuro-symbolic Euclidean geometry; ~100M synthetic theorems; LM proposes auxiliary constructions; symbolic engine deducts. **25/30** IMO-level geometry problems vs previous 10; average human gold ~25.9 on that test [S30][S31]. Proofs are in a geometry DSL, human-readable, **machine-checkable in that engine**.

**AlphaGeometry 2 (sourced).** Chervonyi, Trinh et al., arXiv:2502.03544 / JMLR 2025: language coverage of IMO 2000–2024 geometry 66%→88%; solving rate **84%** of those geometry problems vs 54% previously; part of the IMO 2024 silver-medal **pipeline** [S32].

**AlphaProof + AG2 at IMO 2024 (sourced).** DeepMind blog 25 July 2024 (updated note 12 November 2025 pointing to a *Nature* methodology paper): **4 of 6** IMO 2024 problems, **28/42**, silver-medal band (gold started at 29). AlphaProof: Lean + Gemini autoformalization of a problem bank + AlphaZero-style RL; **problems were manually formalized** for the contest; **P1, P2, P6** solved in Lean after **two to three days** of test-time RL; combinatorics **P3, P5 unsolved**; AG2 solved geometry **P4** in 19 seconds after formalization. Gemini 1.5 Pro with tools was used to **guess answers** for “find all…” problems; AlphaProof **refuted** most wrong candidates [S33]. Hubert et al., *Nature* 2025, restates this protocol [S34].

**IMO 2025 Gemini Deep Think (sourced).** DeepMind blog: an advanced Gemini produced natural-language proofs inside the 4.5-hour contest window and was **officially graded** by IMO coordinators; they contrast this with 2024’s Lean translation and multi-day compute [S35]. Classification: **contest performance**, not a research-problem resolution. Informal proofs still need human grading unless separately formalized.

**Mechanism that actually worked (sourced synthesis).** Generate many candidates → **reject** with Lean / DD+AR / an evaluator → reinforce what verified. Scale helps **proposal**; the **small exact kernel** is what converts hallucination into a theorem. Autoformalization remains a bottleneck (2024 IMO: humans still wrote Lean statements).

### 1.4 Negative results and hype corrections (sourced)

| Episode | What was actually true | Lesson |
| --- | --- | --- |
| PINNs as universal PDE solvers | Fail on moderately hard convection/reaction without curriculum [S11] | Soft constraints ≠ hard physics |
| FunSearch marketing | New **constructions**, not a closed-form cap-set theorem [S20] | Name the object improved |
| AlphaTensor | Better **small-matrix** algorithms, not \(\omega\) [S23] | Local optimum ≠ open-problem kill |
| GPT-5 / Erdős (2025 press) | Literature retrieval misread as novelty [S40] | Check Bloom/site status vs actual literature |
| Grok 3 Riemann | Joke [S44] | Ignore social posts without a paper |
| Hauer & Kondrak Voynich Hebrew | Strong synthetic cipher results; Voynich application **suggests** Hebrew under anagram+abjad assumptions; authors note possible artifacts [S50] | Powerful LM + flexible decoder manufactures readings |
| Voynich LMs that generate EVA-like text | Prior work already trains predictors; matching n-grams or surface stats is not meaning ([PRIOR_WORK.md](../PRIOR_WORK.md)) | Generation is not decipherment |

---

## 2. Mechanisms (what the systems actually do)

1. **Surrogate operators.** Learn \(u_0\mapsto u(\cdot,t)\) or residual-minimizing fields. Verifier: numerical error vs a classical solver / observations. **Does not certify existence.**
2. **Physics-informed / hard constraints.** Soft residuals (PINN) vs architecture that **enforces** divergence-free structure, conservation, or a Lean type. Soft losses can be gamed; hard kernels cannot (up to definitional bugs).
3. **Symbolic regression.** Sparse library (SINDy) or recursive symmetry search (AI Feynman). Verifier: residual on held-out trajectories **and** human-readable formula. Collapses if the library is wrong.
4. **Program search (FunSearch / AlphaEvolve).** LLM mutates code; evaluator returns a scalar. Verifier must be **cheap, exact, and ungameable**.
5. **Formal proof search.** Search in Lean/Isabelle/geometry DSL; each successful tactic sequence is a certificate.
6. **Conjecture and counterexample search.** Propose objects; check a predicate. AlphaTensor, cap sets, some Erdős disproofs.
7. **Verifier-in-the-loop.** The robust pattern across FunSearch, AlphaProof, AlphaEvolve, and (claimed) NS Lean: **generation is cheap; acceptance is strict.**
8. **Curriculum and lemma libraries.** Synthetic theorem curricula (AlphaGeometry); Mathlib; test-time RL on problem variants (AlphaProof). Decomposition is real but **domain-specific**.
9. **Where scale helps vs where a tiny verifier matters.** Scale: informal language, proposal diversity, autoformalization quality. Tiny verifier: combinatorics, type-checking, packing scores, “is this a cap set?” **If the predicate is “sounds like Latin,” scale makes self-deception easier.**

---

## 3. Analogy map (math technique → linguistics → Voynich)

Applicability: **high** / **conditional** / **low**. “Conditional” means the analogue is useful only under an explicit extra assumption (known language family, known channel class, synthetic ground truth, etc.).

| Math / scientific mechanism | Linguistic / crypto analogue | Voynich applicability | Why |
| --- | --- | --- | --- |
| Neural PDE **surrogate** | Neural language model of EVA | **Low** as a decipherment method | Predicts the **observation process**, like FNO predicting a flow field. EXP-0002 already shows compact LMs beat n-grams on development data; EXP-0010 failed longer context. A better EVA generator is not a reading. |
| PINN **soft** residual | “Looks like language / Zipf / morphology” auxiliary losses | **Low** | Krishnapriyan: soft constraints are optimizable without truth. Structured gibberish can match selected stats (project HYP-005; Bowern contested). |
| **Hard** constraint (divergence-free architecture, Lean kernel) | Typed channel: substitution, homophony, nulls, copy-mutate with **executable** decoder | **High** (on synthetics first) | Copiale succeeded when the **channel hypothesis** became checkable (Roman letters = spaces, colon = gemination) [S51]. Protocol already requires frozen mappings. |
| Operator learning (DeepONet/FNO) | Map “key/language → ciphertext law” across **families of generators** | **Conditional** | Useful as **meta-learning of channels** (NEXT_DESIGN P1–P2), not as a NS-style continuum. EXP-0008 already showed familiar-key gains without unfamiliar-key transfer. |
| SINDy / AI Feynman | Induce a **small** morphology/phonology/rule set from counts | **Conditional** | Works if the “library” (affixes, copy edits, null alphabet) is right. Goldwater/Griffiths/Johnson Bayesian segmentation shows context assumptions change the recovered units [S52]. Wrong library → confident wrong grammar. |
| FunSearch program search | Search **decoder programs** scored by held-out likelihood **plus** independent constraints | **High** if the score is not “fluent English” | Same as Copiale’s hypothesize-and-retract loop, automated. Without a hard score, this becomes Gibbs-style Latin. |
| AlphaTensor tensor game | Search **keys** in a finite cipher family with exact encrypt/decrypt | **Conditional** | High for known-family cryptanalysis (Knight 2006-style attacks [S51]); low if the family is unknown. |
| AlphaGeometry synthetic curriculum | Mass synthetic **known** encodings (already EXP-0011–0016 direction) | **High** as a **method test**, **low** as immediate manuscript meaning | Linear B is a method precedent: constrained hypotheses + **independent names/ideograms**, not neural fluency (Ventris & Chadwick 1953 [S53]—abstract/Cambridge pages fetched; full JHS PDF paywalled). |
| AlphaProof Lean loop | Autoformalize? **No good analogue** for unknown script | **Low** for historical meaning; **conditional** for **synthetic** recovery specs | There is no Mathlib of “what the herbal pages mean.” Lean can specify **synthetic** pass rules (already used informally in EXP registrations). |
| Counterexample search | Falsify “this is simple substitution English,” “this is copy-mutate only,” etc. | **High** | The project’s protocol already; AlphaEvolve-style search can propose generators that **fail** preregistered diagnostics. |
| Verifier-in-the-loop | Held-out recon_acc, matched-random, vocab-filter, independent anchors | **High** | EXP-0011–0013: metrics without reconstruction were gamed (delete-nothing F1). Exact-count decode is the local analogue of a kernel. |
| Scale without verifier | Pretrained LLM “translation” of Voynich | **Low / reject** | CipherGAN and unsupervised MT (Conneau/Lample [S54]) assume a **target language corpus**. Hauer–Kondrak shows language ID can fire under false channel assumptions [S50]. |

**Linear B as method, not promise (sourced, limited).** Ventris & Chadwick, *JHS* 73 (1953) “Evidence for Greek Dialect in the Mycenaean Archives” [S53]: systematic sign-group analysis on published Pylos/Knossos corpora, place-name leverage, then Greek morphological testing; later confirmed on unpublished tablets (Antiquity 1953 summary [S53b]). **We did not re-read the full 1953 PDF body in this pass** (Cambridge Core abstract only). Methodological content we **do** treat as sourced from Copiale and from this project’s existing decipherment review: **grid of constraints, independent semantic pegs, prediction on unseen tablets**—not “an LM invented Mycenaean.”

**Where the Navier–Stokes / Erdős analogy fails (strongest breaks).**

1. **Known equation vs unknown generative class.** NS has a named PDE. Voynich may be language, verbose cipher, abbreviation, copy-mutate, mixed, or none of these (HYP-001–006 unresolved).
2. **A/B vs C/D style answers.** Forced NS blowup answers a **specific alternative**. A forced “reading” that inserts unbounded nulls answers a different, easier problem.
3. **Exact finite check vs semantic fluency.** A cap set is yes/no. A Voynich “translation” is infinitely flexible unless the decoder is frozen and tested on held-out leaves.
4. **Energy/physics residual vs Zipf.** Matching a residual of a known operator is not like matching hapax rates.
5. **Mathlib vs empty meaning library.** Formal math reuses lemmas. Decipherment needs **external** anchors (language, genre, images) chosen without peeking at the translation.
6. **Scale of compute that produced the 2026 NS claim** (Nature: thousands of agents, days [S03]) is not a local MPS recipe and is not the bottleneck here; **identifiability** is (EXP-0008, HYP identifiability paragraph).

---

## 4. What this repository should try

**Already done (do not rebuild):** compact EVA LMs and bits/token vs n-grams (EXP-0001/0002); context/shuffle/layout (0003/0005/0007/0010); supervised then label-free synthetic state (0004/0006/0008); causal patching (0009); CTC/null recovery on cipher+filler with Finnish then multilingual scale-up (0011–0016 in progress). Deep-review P1–P5 (canonical keys, edge-emitting HMMs, joint futures) are designed, not executed. World-model/diffusion memos are ideation, not runs.

**Ranked next experiments** (falsifiable; small unless noted). Each is a **working hypothesis** until registered under PROTOCOL.md.

### R1. FunSearch-style **decoder-program** search with a frozen exact score (small local)

- **Hypothesis:** among a **typed** family (delete-nulls + copy/mutate + homophone classes + simple substitution into a **held-out language**), some short program recovers synthetic \(C(L)\) better than matched-random and vocab-filter, transferring across languages/families the way EXP-0016 intends.
- **Verifier:** the existing recon/null gates, plus **program length penalty**, plus no peeking at test ZL3b. Human-readable program required (FunSearch lesson).
- **Support:** FunSearch/AlphaEvolve; Copiale hypothesize-and-retract; Knight et al. 2006 attacks cited in Copiale.
- **Weakeners:** EXP-0011–0013 failed reconstruction with a neural mask; if the **family is wrong**, search will still “win” on a bad score. Unsupervised MT needs a target corpus [S54].
- **Cost:** small–medium local (mutate Python, not 10k agents).
- **Failure mode:** scoring EVA likelihood of a Latin LM (Hauer–Kondrak failure mode).

### R2. Finish the **exact-count / joint keep-run** synthetic gate before any Voynich label-free decode (small local; already highest-EV in NB-0026b)

- **Hypothesis:** EXP-0013 failed because of **decode**, not representation (exploratory exact-count ~0.218 vs 0.199; not a scored pass).
- **Verifier:** frozen EXP-0016 (or 0013) rule; no threshold retune.
- **Support:** AlphaProof’s “guess many answers, refute most.”
- **Cost:** small. **Do not** relaunch architecture search in parallel.

### R3. **Generator-adversary** search for HYP-005 (small local)

- **Hypothesis:** an evolutionary program that emits copy-mutate/self-citation text can be **forced** to match preregistered surface stats **and then fail** a withheld diagnostic (joint futures, line-initial inventory, A/B Jaccard under NB-0022-style rules).
- **Verifier:** preregister **held-out** metrics; matching 4/6 selected stats is not a hoax proof (already in NB-0022).
- **Support:** AlphaEvolve construction search; Timm–Schinner-type generators in existing review.
- **Cost:** small. **Failure mode:** metric shopping.

### R4. Explicit **edge-emitting** state + joint continuations (NEXT_DESIGN P2) (small–medium local)

- **Hypothesis:** a compact probabilistic channel predicts **strings**, not just next tokens, on fresh keys better than the transformer reference.
- **Verifier:** P2 failure conditions already written (probability validity, no unique recovery on equivalent generators).
- **Support:** operator-learning analogue; EXP-0008 negative on naïve transfer.
- **Cost:** medium local. Overlaps designed work—**implement, don’t redesign.**

### R5. **Counterexample search** against simple-language-ID and simple-substitution (small local)

- **Hypothesis:** no language in a frozen UDHR/FLORES panel plus monoalphabetic/homophonic attack (Knight 2006 / Hauer methods) wins on **scrambled-glyph and section-shuffled** controls as well as on true EVA.
- **Verifier:** the same attack must not prefer Hebrew/Latin on **null** text with matched length/Zipf. Hauer already warned of artifacts [S50].
- **Cost:** small. Rejects a whole class without a translation.

### R6. Bounded **LLM-as-proposer** with **local verifier only** (optional; paid API later)

- **Hypothesis:** a frontier model can propose decoder **skeletons** that humans/local search instantiate; it must **not** judge translations.
- **Verifier:** proposals that fail R1/R2 scores are discarded automatically.
- **Support:** AlphaEvolve + Tao: models help when the user supplies the evaluator [S27].
- **Cost:** large/paid if used for sampling; **not required** for R1–R5. User-reported GPT-6 credits exist in MEMORY; **not authorized here as a spend.**

### R7. Image-text **anchors** (BACKLOG / P5) — **defer**

- **Hypothesis:** independently coded motifs constrain a frozen lexicon.
- **Verifier:** hold out leaves; no plant-ID → word circularity (PROTOCOL).
- **Cost:** data/annotation first. **Low until anchors exist.** Copiale used **plaintext German**, not pictures.

---

## 5. Explicit non-recommendations

- **Do not** train a larger EVA LM to “solve it like Navier–Stokes.” FNO/PINN success is surrogate modeling; EXP-0010 already failed matched longer context.
- **Do not** use PINN-style “language residual” losses as evidence of language.
- **Do not** ask an LLM to produce a Voynich translation or a Lean-free millennium-style writeup of a reading.
- **Do not** treat 2026 NS C/D (forced blowup) as license to pick an easy alternative (unbounded filler) and call it decipherment.
- **Do not** rerun Hauer–Kondrak-style anagram+abjad+Hebrew ranking as a confirmatory language ID without null corpora.
- **Do not** autoformalize manuscript meaning. There is no kernel for “this plant is *hellebore*.”
- **Do not** spend frontier-agent-swarm compute until R1–R5 exist; the NS swarm is a **different problem class**.
- **Do not** implement the full world-model/diffusion program as a substitute for a typed decoder (that memo is ideation).

---

## 6. Relation to existing project documents

- Does **not** contradict CAMPAIGN-0001 negatives: unfamiliar-key transfer failed; manuscript causal maps failed; more context did not win.
- Aligns with NEXT_DESIGN: explicit channels, joint futures, independent semantic anchors.
- Aligns with DECIPHERMENT.md: generalization across encodings ≠ fitting one manuscript.
- Updates the MEMORY note that NS was “unverified motivation”: we **have** now sourced the 2026 claim’s **actual statement** (forced C/D-style blowup, prize not awarded). We have **not** independently verified Lean.

---

## 7. Claims considered but not sourced enough to assert

- OpenAI HTML blog details (agent counts, 88 hours, GPT-6 Astra Lean time) appear in Nature and secondary articles; the **blog itself 403’d**. Agent-count figures are **Nature-reported**, not independently audited.
- Alpöge–Buckmaster Euler paper and Anandkumar PINN Euler paper (Nature news): **not fetched**.
- Independent `lake build` of OpenAI Lean: **not run**.
- erdosproblems.com #846/#793 Lean proofs: **snippets only**.
- October 2025 GPT-5 Erdős tweets: **secondary press only**.
- Grok 4.5 \(S^4\) hypercontractivity: **press only; omitted as fact**.
- Full Ventris–Chadwick 1953 article body: **paywall**; method summarized from abstracts plus standard later descriptions, labeled as limited.
- Whether CMI will treat forced C/D as prize-complete: **rules allow either direction**, but institutional acceptance is **future**.
