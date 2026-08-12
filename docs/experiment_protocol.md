# Experiment protocol

This document is the evaluation contract for the project. Changes to it require explicit justification in a pull request.

## Research question

Can response-based knowledge distillation improve Gemma 4 E2B on fine-grained intent classification and out-of-scope detection beyond ordinary supervised fine-tuning?

## Dataset partitions

The official CLINC150 full split is used without modification:

| Partition | In-scope | OOS | Permitted use |
|---|---:|---:|---|
| Train | 15,000 | 100 | Training and synthetic-data seeding |
| Validation | 3,000 | 100 | Model selection and prompt development |
| Test | 4,500 | 1,000 | Final evaluation only |

No test utterance may be included in a teacher prompt, synthetic-data prompt, training set or few-shot demonstration. The original order and labels are retained.

## Phase 1: zero-shot baseline

The instruction-tuned student receives:

1. A fixed system instruction.
2. The complete alphabetically sorted label vocabulary.
3. One user utterance.

Generation is greedy with thinking disabled. The parser accepts a single known label or a simple `intent:`/`label:` wrapper. Unknown, empty and ambiguous generations are counted as invalid predictions and therefore as errors.

The baseline uses no CLINC150 demonstrations, descriptions or teacher outputs.

## Metrics

The primary metric is macro-F1 over the 151 expected classes. Secondary metrics are:

- Overall accuracy.
- In-scope accuracy over the 150 known intents.
- Binary OOS precision, recall and F1.
- Invalid-output rate.
- Top 20 ordered confusion pairs.

Every reported score must identify the Git commit, configuration file, model revision when available, and number of evaluated examples.

## Planned comparisons

| ID | Student training data | Purpose |
|---|---|---|
| ZS | None | Zero-shot baseline |
| SFT | Official train split | Conventional supervised baseline |
| KD-R | Teacher-generated paraphrases | Response distillation |
| KD-E | Teacher corrections of student errors | Error-focused distillation |
| KD-H | Hard negative and boundary examples | Improve confusing intent pairs |

Teacher-generated data will be versioned by prompts, generation parameters and content hashes. It will not be committed if provider terms prohibit redistribution.

