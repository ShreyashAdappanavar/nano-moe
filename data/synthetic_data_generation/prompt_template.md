## Purpose
Define the canonical instruction schema and validation rules for synthetic TinyStories-style alignment data.
This file is the single source of truth for prompt construction and output checks.

---

## Prompt seed schema (prompts_seed.jsonl)
Each line is a JSON object with:

- id: string (unique)
- split: "train" | "val"
- protagonist: string
- theme: string (must be in THEME_SET)
- required: list[string] (len 2..4)
- banned: list[string] (len 0..2)
- min_sentences: int
- max_sentences: int
- instruction: string (must match the template below when rendered)

---

## Global constraints
- Sentence range (default): 6..9 (use per-example min_sentences/max_sentences fields)
- required phrases:
  - len(required) in [2, 4]
  - each required phrase length 3..40 chars
  - required phrases should be concrete objects/short phrases (avoid punctuation-sensitive strings)
- banned phrases:
  - len(banned) in [0, 2]
  - banned phrases length 3..30 chars
- Language: English
- Content: child-friendly; no violence/gore; no sexual content; no hate/harassment; no self-harm

---

## Allowed themes (THEME_SET)
Use one of:
- friendship
- kindness
- honesty
- courage
- curiosity
- sharing
- patience
- teamwork
- responsibility
- gratitude

---

## Canonical instruction template (rendered into `instruction`)
Render exactly with these fields:

Write a children's story.

Constraints:
- Protagonist: {protagonist}
- Theme: {theme}
- Length: {min_sentences} to {max_sentences} sentences.
- Must include ALL of these exact phrases (case-insensitive match is acceptable):
  {required_list}
- Must NOT include any of these phrases (case-insensitive match):
  {banned_list_or_NONE}

Style:
- Simple words and short sentences.
- Child-friendly tone.
- No meta commentary about writing.
- Do not use bullet points or numbered lists.

Formatting:
- Output plain text only.

Rendering rules:
- required_list is rendered as one phrase per line, prefixed with "- ".
- banned_list_or_NONE:
  - if banned is empty, render exactly: "NONE"
  - else render one phrase per line, prefixed with "- ".

---

## Output validation (programmatic)
Given (instruction fields, output_text):

1) Non-empty
- output_text.strip() must be non-empty

2) Sentence count
- Split sentences using regex: /[.!?]+/
- Strip whitespace from segments; drop empty segments
- sentence_count = number of remaining segments
- Pass if min_sentences <= sentence_count <= max_sentences

3) Required phrases present
- For each r in required:
  - lower(r) must be a substring of lower(output_text)

4) Banned phrases absent
- For each b in banned:
  - lower(b) must NOT be a substring of lower(output_text)

5) Length cap (anti-ramble)
- len(output_text) <= 2000 characters

---

## Failure labels (for rejects + DPO)
Allowed labels:
- missing_required: at least one required phrase missing
- contains_banned: at least one banned phrase present
- wrong_sentence_count: sentence_count outside bounds
- too_long: char length cap violated
- other: anything else unexpected