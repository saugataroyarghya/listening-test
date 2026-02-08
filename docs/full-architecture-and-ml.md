# Speech Analysis API: Full Architecture, ML Models, and Algorithms

This document describes the current production behavior of the codebase in detail:

1. Full system architecture and request lifecycle.
2. ML models, deterministic algorithms, metric formulas, prompting, and score fusion.

## 1. Full Architecture

### 1.1 Runtime components

- API layer: `main.py` (FastAPI app, routing, response composition).
- Speech/feature layer: `services.py` -> `WhisperService`.
- LLM rubric layer: `services.py` -> `GroqService`.

### 1.2 Entry points

- `GET /`: basic API metadata.
- `GET /health`: readiness (`whisper_loaded`, `groq_configured`).
- `GET /analyzeSpeech?url=...`: full transcription + analysis pipeline.
- `POST /analyzeSpeechFile`: multipart file-upload analysis pipeline (`file` field, mp3/audio).

### 1.3 External dependencies

- Audio source: arbitrary URL passed to `/analyzeSpeech`.
- STT model: `faster-whisper` (`small.en`, CPU, `int8`).
- LLM scorer: Groq chat completions (`llama-3.3-70b-versatile`).
- Optional deterministic grammar engine: `language_tool_python` (if installed/available).

### 1.4 End-to-end flow (`/analyzeSpeech`)

1. Download audio bytes via `httpx` into temporary `.mp3`.
2. Transcribe with word timestamps and probabilities.
3. Run alignment refinement to stabilize timestamps.
4. Compute local deterministic metrics:
   - speaking speed + pauses + fluency features
   - pronunciation clarity
   - prosody proxies (intonation/stress, chunking/rhythm)
   - lexical resource
   - grammar analysis
5. Send transcript + metrics context + deterministic feature context to LLM.
6. Combine LLM and local scores for fluency, lexical, grammar.
7. Compute pronunciation score from 3 pronunciation subscores.
8. Compute overall band score as mean of available >0 category scores.
9. Return merged JSON response (`transcript`, `words`, `alignment`, `analysis`).

### 1.5 High-level architecture diagram

```mermaid
flowchart TD
  A[Client] --> B[/analyzeSpeech]
  B --> C[Audio download via httpx]
  C --> D[Whisper STT with word_timestamps]
  D --> E[Alignment refinement]
  E --> F1[Fluency and pauses]
  E --> F2[Pronunciation clarity]
  E --> F3[Prosody proxies]
  E --> F4[Lexical analyzer]
  E --> F5[Grammar analyzer]
  E --> G[Transcript and annotated transcript]
  F1 --> H[Groq IELTS LLM scoring]
  F3 --> H
  F4 --> H
  F5 --> H
  G --> H
  H --> I[Hybrid score fusion]
  F1 --> I
  F2 --> I
  F3 --> I
  F4 --> I
  F5 --> I
  I --> J[Final JSON response]
```

## 2. ML Models and Algorithms

## 2.1 STT model and transcript extraction

- Model: `WhisperModel("small.en", device="cpu", compute_type="int8")`.
- Transcription options:
  - `language="en"`
  - `word_timestamps=True`
  - `vad_filter=False`
  - `initial_prompt="Um, uh, like, you know, hmm, ah"` (preserve fillers)
- For each word:
  - text: `word.word.strip()`
  - confidence: `word.probability` (if `NaN`, replaced with `0.5`; rounded to 2 decimals)
  - timestamps: rounded `start`/`end` (2 decimals)

Outputs:
- `text`: joined token sequence
- `annotated`: `word(confidence)` sequence
- `words`: list of `{word, confidence, start, end}`

## 2.2 Alignment refinement algorithm

Function: `refine_word_alignment(words_data)`.

Purpose:
- enforce monotonic boundaries
- fix overlaps and impossible durations
- smooth micro-gaps

Key rules:

- Estimated fallback duration:
  - `estimated_duration = max(0.08, min(0.45, len(word) * 0.03))`
- If `0 < gap < 0.04` between previous end and current start:
  - snap `start = previous_end` (`micro_gap_fixes += 1`)
- If `gap < 0`:
  - set `start = previous_end` (`overlap_fixes += 1`)
- If `end <= start`:
  - set `end = start + estimated_duration` (`duration_fixes += 1`)
- If `(end - start) < 0.06`:
  - enforce `end = start + 0.06` (`duration_fixes += 1`)

Alignment metadata returned:
- `adjusted_words`
- `overlap_fixes`
- `duration_fixes`
- `micro_gap_fixes`

## 2.3 Fluency and coherence metrics (deterministic)

Function: `calculate_speaking_metrics(words_data)`.

### Core timing metrics

- Total duration:
  - `total_duration = last_word_end - first_word_start`
- Words per minute:
  - `wpm = round((total_words / total_duration) * 60)` if `total_duration > 0`
- Speech time:
  - `speech_time = sum(max(0, end - start) for each word)`
- Articulation rate:
  - `articulation_rate_wpm = (total_words / speech_time) * 60`
- Phonation time ratio:
  - `phonation_ratio = speech_time / total_duration`

### Pause analysis

- Pause candidate if gap `> 0.3s`.
- Good pause if `0.3 <= gap <= 1.5`.
- Bad pause if `gap > 1.5`.
- Pause percentiles:
  - `pause_p50` and `pause_p90` via interpolated percentile over pause durations.

### Disfluency proxies

- Filled pauses:
  - single-token set: `um, uh, hmm, ah, er, like, basically, actually, well`
  - phrase set: `you know`
- Filled pause rate:
  - `filled_pause_rate_per_min = filler_count / (total_duration / 60)`
- Repetitions:
  - count adjacent repeated normalized tokens
- Self-correction markers:
  - occurrences of `i mean, sorry, rather, or rather, that is`
- Cohesive devices:
  - count occurrences of connectors (e.g., `however`, `therefore`, `because`, `for example`, `on the other hand`)

### Mean length of run

- Split runs when inter-word gap `> 0.25s`.
- `mean_length_of_run_words = average(run_lengths)`.

### Local fluency score formula (1-9 clamped)

- Base:
  - `score_base = 9.0`
- Penalty terms:
  - WPM outlier penalty:
    - `+1.8` if `wpm < 95` or `wpm > 190`
    - else `+0.8` if `wpm < 110` or `wpm > 175`
  - bad pause penalty:
    - `+min(2.0, bad_pause_count * 0.5)`
  - filler penalty:
    - `+min(1.2, filled_pause_rate_per_min / 8)`
  - repetition penalty:
    - `+min(1.0, repetition_count * 0.3)`
  - low phonation penalty:
    - `+0.8` if `phonation_ratio < 0.55`
  - short-run penalty:
    - `+0.7` if `mean_length_of_run_words < 3`
- Final:
  - `fluency_score = clamp_1_to_9(score_base - total_penalty)`

## 2.4 Pronunciation metrics

## 2.4.1 Clarity score from STT confidence

Function: `calculate_pronunciation_clarity(words_data)`.

Token bins (filler words excluded):
- clear: `confidence >= 0.85`
- acceptable: `0.7 <= confidence < 0.85`
- unclear: `confidence < 0.7`

Core formula:
- `clarity_percentage = ((clear + acceptable) / total_assessed) * 100`

Band mapping:
- `>=95 -> 9`
- `>=90 -> 8`
- `>=85 -> 7`
- `>=80 -> 6`
- `>=70 -> 5`
- `>=60 -> 4`
- `>=50 -> 3`
- else `2`

Also returns:
- `average_confidence`
- `word_counts`
- top 10 worst-confidence `unclear_words`

## 2.4.2 Intonation/stress and chunking/rhythm proxies

Function: `calculate_prosody_features(words_data)`.

Derived variables:
- word duration vector
- inter-word gap vector
- `duration_cv = pstdev(durations)/mean(durations)`
- `gap_cv = pstdev(gaps)/mean(gaps)` when mean gap > 0
- chunk lengths split on gaps `> 0.35`
- `chunk_cv = pstdev(chunk_lengths)/mean(chunk_lengths)`
- long pauses count: gaps `> 1.5`

Intonation/stress score (1-9 clamped):
- base `8.5`, penalty:
  - `+1.5` if `duration_cv < 0.15`
  - `+0.9` if `duration_cv > 0.75`
  - `+0.8` if `avg_confidence < 0.72`
  - `+0.8` if `gap_cv > 1.6`
- `intonation_score = clamp_1_to_9(8.5 - penalty)`

Chunking/rhythm score (1-9 clamped):
- base `8.5`, penalty:
  - `+1.2` if `mean_chunk_length < 2.5`
  - `+0.8` if `mean_chunk_length > 12`
  - `+1.0` if `chunk_cv > 0.85`
  - `+min(1.5, long_pause_count * 0.4)`
- `rhythm_score = clamp_1_to_9(8.5 - penalty)`

Pronunciation overall category score in API:
- `pronunciation_score = mean(clarity_score, intonation_score, rhythm_score)`

## 2.5 Lexical resource model (local deterministic)

Function: `calculate_lexical_resource(transcript, words_data)`.

Tokenization:
- regex: `[a-zA-Z']+`, lowercase.

Features:
- `ttr = unique_tokens / total_tokens`
- moving TTR window:
  - window size = `min(25, total_tokens)`
  - `moving_ttr = mean(TTR(window_i))`
- sophisticated ratio:
  - content words with `len(word) >= 7` over content words
- CEFR lexicon matching:
  - deterministic lexicon tiers: `A1, A2, B1, B2, C1`
  - token mapped to the highest tier where present
  - `coverage = matched_content_tokens / total_content_tokens`
  - `difficulty_score = mean(CEFR_WEIGHT(token_level))`, with `A1=1 ... C1=5`
  - `unknown_ratio = unmatched_content_tokens / total_content_tokens`
- idiom hits from phrase list:
  - `"on the other hand"`, `"as far as i know"`, `"at the end of the day"`, `"in my opinion"`, `"for instance"`, `"to be honest"`, `"as a result"`
- repetition ratio:
  - `max(token_frequency) / total_tokens`
- optional average confidence from `words_data`

Scores (all clamped to 1-9):

- Vocabulary range:
  - `cefr_weighted_range = 4.2 + (difficulty_score * 0.95)`
  - `diversity_range = 2.4 + (moving_ttr * 5.1) + (sophisticated_ratio * 3.1)`
  - `range_score = 0.65*cefr_weighted_range + 0.35*diversity_range`
- Accuracy:
  - start `8.0`
  - penalty `+1.2` if `repetition_ratio > 0.14`
  - else penalty `+0.6` if `repetition_ratio > 0.1`
  - additional `+0.7` if `avg_confidence < 0.72`
  - additional `+0.5` if `unknown_ratio > 0.75`
  - `accuracy_score = clamp_1_to_9(8.0 - penalty)`
- Idiomatic:
  - `idiomatic_score = clamp_1_to_9(5.5 + min(2.8, idiom_count * 0.8))`
- Overall lexical:
  - `overall_lexical_score = mean(range_score, accuracy_score, idiomatic_score)`
- Deterministic confidence:
  - `deterministic_confidence = clamp_0_to_1(0.7*coverage + 0.3*(1-repetition_ratio))`
- CEFR mapping:
  - `C1` if `difficulty_score >= 4.6` and `coverage >= 0.18`
  - `B2` if `difficulty_score >= 3.6` and `coverage >= 0.22`
  - `B1` if `difficulty_score >= 2.6` and `coverage >= 0.28`
  - `A2` if `difficulty_score >= 1.8` and `coverage >= 0.32`
  - else `A1`

Returned local lexical metadata includes:
- `deterministic_confidence`
- `cefr_coverage`, `cefr_difficulty_score`, `cefr_unknown_ratio`
- `cefr_level_counts`
- `engine = "cefr_lexicon"`

## 2.6 Grammar model (local deterministic)

Function: `calculate_grammar_analysis(transcript)`.

Features:
- sentence split on `[.!?]+`
- complex sentence ratio:
  - proportion of sentences containing subordinators:
    - `because, although, while, whereas, unless, since, if, when, after, before`
- clause marker count
- heuristic error detectors (always-on):
  - `i is`
  - `(he|she|it) are|have`
  - `a + vowel-start word`
  - `an + consonant-start word`
  - `did + past-tense (-ed) form`
  - adjacent repeated words (`word word`)
- optional `LanguageTool` pass:
  - if `language_tool_python` is available and initializes successfully, additional deterministic grammar matches are merged
  - if unavailable/fails, fallback remains heuristic-only
- max returned errors: 16

Scores (clamped 1-9):

- Structure range:
  - `range_score = 4.5 + (complex_sentence_ratio * 4.0) + min(1.5, clause_marker_count * 0.25)`
- Grammar accuracy:
  - `error_density = (error_count / token_count) * 100`
  - `accuracy_penalty = min(4.2, error_density / 5.8)`
  - if LanguageTool active: `accuracy_penalty += min(1.2, language_tool_match_count / 10)`
  - `accuracy_score = 8.5 - accuracy_penalty`
- Tense accuracy:
  - past markers:
    - `{was, were, had, did, went, made, saw}` or tokens ending in `ed`
  - present markers:
    - `{is, are, am, do, does, have, has, go, goes}`
  - `tense_consistency = max(past_count, present_count) / (past_count + present_count)` (or `1.0` if no markers)
  - `tense_score = 4.0 + (tense_consistency * 5.0)`
- Overall grammar:
  - `overall_grammar_score = mean(range_score, accuracy_score, tense_score)`
- Deterministic confidence:
  - default heuristic-only baseline: `0.55`
  - if LanguageTool active: `min(1.0, 0.75 + min(0.2, language_tool_match_count / 80))`

Returned local grammar metadata includes:
- `deterministic_confidence`
- `language_tool_match_count`
- `engine = "language_tool_plus_heuristics"` or `"heuristic_only"`

## 3. LLM Prompting and Hybrid Scoring

## 3.1 Prompting strategy (`GroqService.get_ielts_analysis`)

Model and parameters:
- model: `llama-3.3-70b-versatile`
- `response_format={"type":"json_object"}`
- `temperature=0.5`
- `max_tokens=2000`

Prompt layers:
- System prompt:
  - fixed IELTS examiner role
  - strict JSON schema with required sections:
    - `fluency_and_coherence`
    - `lexical_resource`
    - `grammar`
    - `filler_words`
    - overall LLM scores
  - includes IELTS band rubric and fluency rubric guidance
- User prompt:
  - transcript
  - annotated transcript with confidences
  - speaking metrics context (WPM, pauses, etc.)
  - deterministic feature context (fluency/prosody/lexical/grammar local metrics)
  - instruction to explain deviations from feature estimates

## 3.2 Hybrid score fusion in API

For fluency, lexical, grammar:

- `combine_scores(llm_score, local_score, llm_weight=0.6)`
- fluency uses fixed blend:
  - `llm_weight = 0.6`
- lexical uses adaptive blend:
  - if `lexical_deterministic_confidence < 0.5`, `llm_weight = 0.7`
  - else `llm_weight = 0.5`
- grammar uses adaptive blend:
  - if `grammar_deterministic_confidence < 0.7`, `llm_weight = 0.65`
  - else `llm_weight = 0.45`
- if both available:
  - `hybrid = llm_score*llm_weight + local_score*(1-llm_weight)`
- if one is missing/zero:
  - fallback to available score

Pronunciation category:
- no LLM blend currently
- deterministic average:
  - `mean(clarity, intonation_and_stress, chunking_and_rhythm)`

Overall band:
- `band_score = mean([fluency, lexical, grammar, pronunciation] where score > 0)`

## 4. Returned response structure

Top-level:
- `transcript`
- `words`
- `alignment`
- `analysis`

`analysis` includes:
- `fluency_and_coherence` (LLM fields + local timing/fluency features)
- `lexical_resource` (`llm` + `local`)
- `grammar` (`llm` + `local`)
- `pronunciation` (clarity + prosody proxies)
- `overall` (hybrid scores and final band)
- `filler_words`
- `low_confidence_words`
- `average_confidence`

## 5. Notes and limitations

- Prosody currently uses timing/confidence proxies, not direct F0 extraction.
- Lexical scoring is now deterministic CEFR-lexicon based, but lexicon breadth still limits coverage and domain vocabulary handling.
- Grammar scoring is deterministic with optional LanguageTool; behavior degrades gracefully to heuristics when the tool is unavailable.
- LLM output stability still depends on prompt/model behavior.
- Overall band is an internal composite score and is not externally calibrated to official IELTS examiner labels.
