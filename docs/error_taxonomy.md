# Error taxonomy for failed retrievals (S2-6)

- **Version:** 1.0
- **Date:** 2026-11-27
- **Status:** frozen before annotation. Changes after annotation starts must be added as a dated changelog entry at the bottom of this file, never silently.

## 1. Purpose and unit of annotation

The taxonomy explains why the best dev system fails on a post. A **failure** is a dev post for which no gold fact-check (FC) is in the system's top 10. **One failed post = one case.**

For each case the annotator sees (local-only export, never committed):

- the post: original text, English translation, and OCR text of attached images;
- the gold fact-check(s): claim + title;
- the system's top-10 retrieved fact-checks with their ranks;
- a precomputed `truncated` flag and the post's token count (for TRUNC).

Annotators assign exactly one PRIMARY label and optionally SECONDARY labels (see section 3).

## 2. Categories

The examples are invented paraphrases, not taken from the dataset.

### IMG: image-dependent
- **Definition:** the claim is carried only by an image or video and is not recoverable from the post text plus OCR.
- **Assign when:** the text/OCR does not state the claim, refers to the media ("look at this", "see the video"), and the gold FC is evidently about the visual content.
- **Do not assign when:** the claim is readable from text or OCR, even if poorly (then TRANS or another label); or the text is simply too short with no sign of relevant media (then WEAK).
- **Example:** post text "Unbelievable, share before it is deleted!" with an empty OCR; the gold FC debunks a doctored photo of a politician at a protest.

### MULTI: multiple claims
- **Definition:** the post contains several distinct claims, and the retrieved fact-checks match a different claim from the one the gold FC covers.
- **Assign when:** at least two separable claims are present and the top 10 are relevant to a claim other than the gold one.
- **Do not assign when:** there is one claim phrased in several sentences; or the top 10 match no claim of the post (then TOPIC / NUMDATE).
- **Example:** a post says a vaccine contains microchips and also that a city banned cash; the gold FC is about the cash ban, the top 10 are about vaccines.

### TRANS: translation or OCR damage
- **Definition:** machine translation or OCR garbling changes the meaning or loses the key entity, so the system cannot match.
- **Assign when:** comparing the original text and the English translation (or OCR) shows a key name, term, or negation distorted or lost, and this plausibly explains the miss.
- **Do not assign when:** the translation is merely awkward but keeps the key content.
- **Example:** the original names a specific town, the translation renders it as a common noun, and the gold FC is about that town.

### TRUNC: beyond the encoder limit
- **Definition:** decisive information lies after the encoder's 512-token limit.
- **Assign when:** the `truncated` flag is true and the decisive content for matching the gold FC is in the cut part.
- **Do not assign when:** the flag is false; or the flag is true but the decisive content is within the first 512 tokens (the cut part is irrelevant).
- **Example:** a long chain message whose first paragraphs are greetings and whose actual claim about a new tax appears at the end.

### NEARDUP: equally valid match in the top 10
- **Definition:** a top-10 FC is a near-duplicate of, or an equally valid match to, the gold FC. The system was effectively right (label noise).
- **Assign when:** a retrieved FC, in the annotator's judgement, addresses the same claim as the post and could have served as gold.
- **Do not assign when:** the retrieved FC only shares the topic or differs in a decisive detail (then TOPIC / NUMDATE).
- **Example:** gold FC "No, the mayor did not resign"; a top-10 FC from another outlet titled "Mayor has not stepped down, rumour false".

### NUMDATE: same event, different decisive detail
- **Definition:** retrieved FCs concern the same event but differ in a number, date, place or person on which the gold FC depends.
- **Assign when:** the top 10 clearly cover the same event/story, and the gold FC differs from them in one of those details that the post specifies.
- **Do not assign when:** there is no such mismatch, only topical similarity (then TOPIC); or the retrieved FC is equally valid (then NEARDUP).
- **Example:** the post claims a flood killed 40 people in one region; the top 10 debunk a similar claim about another region.

### TOPIC: same topic, different claim
- **Definition:** retrieved FCs share the topic but address a different specific claim, with no number/date/entity mismatch that would make it NUMDATE.
- **Assign when:** the top 10 are topically related (the same disease, election, conflict) but assert or debunk other claims.
- **Do not assign when:** a number/date/place/person mismatch is the difference (then NUMDATE).
- **Example:** the post says a herbal tea cures a virus; the top 10 debunk other supposed cures for the same virus; the gold FC is about the tea.

### WEAK: too little content
- **Definition:** the post has too little content (very short, only a link, hashtags, or emoji) to identify the claim.
- **Assign when:** after reading text and OCR, no claim can be inferred and there is no sign of media (otherwise IMG).
- **Do not assign when:** a claim is identifiable although short.
- **Example:** a post consisting of a hashtag and a shortened link.

### OTHER
- **Definition:** none of the above fits.
- **Assign when:** all other categories were considered and rejected. A free-text note (paraphrased, no quoted data text) is **required**.

## 3. Precedence for the PRIMARY label

When several categories apply, the PRIMARY label is the first one in this order:

**NEARDUP > WEAK > IMG > TRUNC > TRANS > MULTI > NUMDATE > TOPIC > OTHER**

Justification: NEARDUP means the system was effectively right, so it overrides everything. WEAK, IMG, TRUNC and TRANS are input-side limits, where the information needed is missing or damaged before matching, and they outrank matching-side confusions; among them, the more fundamental absence of information ranks higher. MULTI explains why a different valid claim was retrieved. NUMDATE is more specific than TOPIC, and OTHER is the fallback.

Other applicable categories may be listed as optional SECONDARY labels (multi-valued). SECONDARY labels are not used for Cohen's kappa.

## 4. Annotation protocol

- Two annotators label every case independently, blind to each other's labels and to any discussion of the cases.
- Each annotator writes `annotations_<annotator>.csv` locally with columns `case_id, primary, secondary, note`. `secondary` is a `;`-separated list (may be empty). `note` is a short paraphrase and must not quote data text.
- Annotation files with only ids, labels and short paraphrased notes may be committed. The case export (with text) is local-only, lives under `data/`, and is gitignored.
- Cases are shown in a fixed random order, without the other annotator's output.

## 5. Agreement and adjudication

- Cohen's kappa on PRIMARY labels: overall, and per category (one-vs-rest).
- Confusion matrix of the disagreements (annotator A vs B).
- Disagreements are adjudicated by discussion between the two annotators; the final label is recorded in a separate file (`adjudicated.csv`: case_id, final_primary, rationale). The original annotation files are left unchanged.

## 6. Sampling plan

- 150-200 failures from the **analysis** half of dev (split in `configs/splits/dev_error_analysis_split.json`, created by `scripts/make_dev_split.py`).
- Stratified by language: the monolingual languages plus crosslingual (proportional allocation with a minimum per stratum where available).
- Failures come from the best dev system at the time of sampling (reference: `results/results.md`); the system and its run name are recorded with the sample.
- The **held-out** half is never inspected and is used only to confirm the fix.

## 7. Fix-validation plan

- Pick the most frequent addressable category in the adjudicated labels.
- Propose a fix targeted at it and fix its configuration before looking at held-out results.
- Evaluate S@10 and MRR on the held-out half (both tracks where applicable), against the unchanged system on the same half.
- Report the result honestly, including when it does not help.

## Changelog

- 1.0 (2026-11-27): initial version, frozen before annotation.
