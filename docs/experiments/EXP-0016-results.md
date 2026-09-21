# EXP-0016 results — Thousand-language family-holdout exact-count recovery

Registration: `docs/experiments/EXP-0016.md`. Pass rule frozen before scores (`d704e59`). **Not** a Voynich decipherment. EXP-0015 superseded as session bar.

## Setup

- **Pool:** 1,063 romanized varieties (619 ISO 639-3) from UDHR (492; Unicode/OHCHR attribution), FLORES-200 `dev` (196; CC BY-SA 4.0), Tatoeba (356; CC BY 2.0 FR), EXP-0015 salvage (19). Wiki fill not required after variety IDs met the bar (`wiki_added=0`).
- **Skipped (57):** 33 CJK no romanizer; 19 other unromanizable; 5 too short. Not silently dropped from ambition.
- **Romanization:** Latin inventory `a–z` + space; script tables in `romanize.py` + Unidecode for other non-CJK scripts; CJK skipped.
- **Holdout:** 270 varieties; **entire families** unseen: Austronesian (100), Afro-Asiatic (66), Uralic (34), Quechuan (14), Dravidian (14), Mayan (12), Otomanguean (8), isolate (6), Tupian (5), Kartvelian (5), Basque (4), Aymaran (2). Finnish/Hungarian in Uralic.
- **Train:** stratified subset of **400** languages from all non-held-out families (seed 4016); 12,000 train / 800 val; fillers `random_char`+`periodic` @ 0.30.
- **Model:** BiLSTM emb=192, hidden=384, layers=2; **5,440,829** params; MPS; 4,000 updates (~14 min train wall).
- **Decoder (frozen):** exact-count `n_keep=round((1-0.30)*L)`.
- **Command:** `.venv/bin/python -m voynich.mass_lang_recovery --root . --device mps --n-train 12000 --n-val 800 --n-holdout 40 --updates 4000 --filler-families random_char,periodic --no-wiki-fill`
- Clean freeze SHA: `d704e59`. Seeds: data 4016 / model 42 / holdout 4017.

## Pass rule (frozen)

≥100 holdouts beat own matched-random recon; ≥5 families among passers; ≥50% non-IE passers; macro recon over **all** holdouts > macro matched-random; macro null_rec≥0.50, null_prec≥0.50, pred_null∈[0.15,0.45]; vocab-filter macro recon must not beat model.

## Decision: **PASS**

| Macro (all 270 holdouts) | Value |
| --- | ---: |
| recon_acc | **0.3155** |
| matched-random recon | 0.1918 |
| null_recall | 0.7195 |
| null_precision | 0.7022 |
| pred_null_rate | 0.2969 |
| vocab-filter recon | 0.0585 |
| langs beat matched-random | **270 / 270** |
| passer families | **12** (all non-IE) |
| non-IE passer fraction | **1.0** |

## Family table

| Family | n | beat rand | macro recon | macro rand |
| --- | ---: | ---: | ---: | ---: |
| Austronesian | 100 | 100 | 0.325 | 0.194 |
| Afro-Asiatic | 66 | 66 | 0.311 | 0.188 |
| Uralic | 34 | 34 | 0.309 | 0.187 |
| Quechuan | 14 | 14 | 0.282 | 0.179 |
| Dravidian | 14 | 14 | 0.305 | 0.201 |
| Mayan | 12 | 12 | 0.316 | 0.190 |
| Otomanguean | 8 | 8 | 0.334 | 0.212 |
| isolate | 6 | 6 | 0.332 | 0.197 |
| Tupian | 5 | 5 | 0.347 | 0.223 |
| Kartvelian | 5 | 5 | 0.281 | 0.180 |
| Basque | 4 | 4 | 0.322 | 0.195 |
| Aymaran | 2 | 2 | 0.270 | 0.176 |

Full per-language JSON: `results/EXP-0016/per_language_compact.json`, `results/EXP-0016/results.json`.

## Historical transfer (score-only; not in pass rule)

13 Latin / Italian / German varieties; macro recon **0.314** vs matched-random **0.183** (13/13 beat random). Not evidence of Voynich-century language ID.

## EXP-0016b

`copy_mutate` falsifier **PASS** (264/270 beat random; macro recon 0.255 > 0.192). See `EXP-0016b-results.md`.

## Voynich label-free (ZL3b validation only)

After both PASSes. Decoder: exact-count @ assumed null rate **0.30**. **ZL3b test unscored.**

| Metric (208 windows / 24 val pages) | Neural | Classical sticky |
| --- | ---: | ---: |
| mean bits-gain | −0.073 | +0.002 |
| mean matched-random gain | −0.049 | −0.019 |
| fraction beats random (+0.05) | 0.250 | 0.317 |
| mean pred_null_rate | 0.297 | 0.111 |

**Negative structure transfer.** 30% null is an assumption. Positive synthetic recovery ≠ manuscript plaintext. **Not a decipherment.**

Artifact: `results/EXP-0016/voynich_label_free.json`.

## Next change (single) and falsifier

Synthetic family-holdout gate cleared; Voynich structure under fixed 0.30 null did not. **Next:** preregister a discrete null-rate grid on ZL3b **validation only** (exact-count at each rate; same checkpoint); pick the rate that maximizes mean bits-gain over matched-random at that same rate; freeze the rate before any page cherry-pick. **Falsifier:** selected rate still has mean_gain ≤ mean_random_gain, or fraction_beats_random ≤ 0.50. If falsified, do **not** retune architecture — reject fixed-rate filler transfer and escalate a rate-free / joint generative hypothesis under a new experiment id.
