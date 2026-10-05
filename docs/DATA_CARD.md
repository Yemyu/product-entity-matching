# Data card

**English** | [简体中文](zh-CN/DATA_CARD.md)

## Source, roles and scope

The experiment uses a custom frozen set of Walmart–Amazon product pairs. Its role names are project assignments, rather than evidence that an official source test file remained untouched.

| Role | Pairs | Use |
| --- | ---: | --- |
| `fit` | 3,738 | Fit title IDF, tabular references and neural models |
| `dev` | 199 | Select one common epoch for the three-seed ensemble |
| `calibration` | 209 | Select the F1 cutoff |
| `evaluation` | 555 | Report saved fixed-pool results; 190 matches and 365 nonmatches |

The 555-pair evaluation pool is retrospective. Its historical exposure record is incomplete, and an independently fresh or entity-disjoint test split has not been established. Interface checks keep evaluation labels out of prediction inputs, but cannot reconstruct that history.

## Fields and normalization

Business fields are `title`, `brand`, `description`, `price` and `priceCurrency`. Native fields are `modelno` and `category`. `data.business_record(raw)` and `data.native_record(raw)` create the explicit views accepted by the feature pipeline.

Text normalization applies Unicode normalization, case folding and whitespace cleanup. Missing values remain missing. A plain nonnegative decimal can be parsed as a price; an ambiguous currency symbol or separator is retained as an unresolved quality state. Price similarity is enabled only when both prices parse and both currency codes are recognized and equal. A missing or incomparable price has a separate indicator and is not treated as an observed zero price.

IDF (inverse document frequency) assigns higher weights to terms found in fewer training titles, reducing the influence of common words on similarity. Title IDF is fitted on business records deduplicated by content within `fit`. The saved state is reused for every other role. IDs and labels are used for alignment and fitting where appropriate, but are not feature values. Model-like title codes are compared separately from native model numbers, including letter/digit disagreements and cross-field matches.

## Feature dictionary

The order below is fixed by [FEATURES.json](../reproducibility/FEATURES.json). All 42 values are finite and lie in [0, 1]. Similarity values for missing inputs are zero, with presence indicators where specified. The four groups are business comparisons (1–14), fit-only title evidence (15–22), title model-code comparisons (23–30), and native model/category comparisons (31–42). The title-evidence group includes both IDF-weighted and direct comparisons.

| Position | Identifier | Meaning |
| --- | --- | --- |
| 1 | `title_token_jaccard` | Jaccard overlap of title tokens |
| 2 | `title_char_jaccard` | Jaccard overlap of title characters, excluding spaces |
| 3 | `description_token_jaccard` | Jaccard overlap of description tokens |
| 4 | `description_both_present` | Both descriptions contain tokens |
| 5 | `brand_equal` | Both brands are present and equal |
| 6 | `brand_both_present` | Both brands are present |
| 7 | `brand_missing_any` | At least one brand is missing |
| 8 | `numeric_token_jaccard` | Jaccard overlap of numeric tokens in title and description |
| 9 | `numeric_tokens_both_present` | Both records contain numeric tokens |
| 10 | `title_length_ratio` | Smaller/larger count of distinct title tokens |
| 11 | `price_both_parsed` | Both prices were parsed |
| 12 | `price_currency_equal` | Both normalized currencies are present and equal |
| 13 | `price_comparable` | Parsed prices with the same recognized currency code |
| 14 | `price_similarity` | 1 / (1 + absolute log1p price difference), if comparable |
| 15 | `title_word_idf_cosine` | Cosine similarity weighted by fit title-word IDF |
| 16 | `title_word_idf_containment` | Shared IDF-weighted energy / smaller title-word energy |
| 17 | `title_char3_idf_cosine` | Fit-IDF cosine similarity of title character trigrams |
| 18 | `title_compact_char3_idf_cosine` | Fit-IDF cosine similarity of compact-title trigrams |
| 19 | `title_alphanumeric_jaccard` | Jaccard overlap of mixed letter-and-digit title tokens |
| 20 | `title_number_jaccard` | Jaccard overlap of title numeric tokens |
| 21 | `title_numbers_both_present` | Both titles contain numeric tokens |
| 22 | `title_compact_exact` | Nonempty compact titles are equal |
| 23 | `code_left_missing_or_right_missing` | At least one title lacks a model-like code token |
| 24 | `code_both_present` | Both titles contain model-like code tokens |
| 25 | `code_exact_overlap` | At least one extracted code is shared |
| 26 | `code_set_jaccard` | Jaccard overlap of extracted code sets |
| 27 | `code_best_edit_similarity` | Best normalized edit similarity across code pairs |
| 28 | `code_same_letters_different_digits` | A code pair has equal letters and different digits |
| 29 | `code_same_digits_different_letters` | A code pair has equal digits and different letters |
| 30 | `code_one_edit_difference` | An unequal code pair of length at least four differs by one edit |
| 31 | `native_model_missing_any` | At least one native model number is missing |
| 32 | `native_model_both_present` | Both native model numbers are present |
| 33 | `native_model_compact_exact` | Nonempty compact native model numbers are equal |
| 34 | `native_model_edit_similarity` | Normalized edit similarity of native model numbers |
| 35 | `native_model_char3_jaccard` | Jaccard overlap of native-model character trigrams |
| 36 | `native_model_same_letters_different_digits` | Native model numbers share letters but differ in digits |
| 37 | `native_model_same_digits_different_letters` | Native model numbers share digits but differ in letters |
| 38 | `cross_native_title_any` | Either native model appears in the opposite title code set |
| 39 | `cross_native_title_both` | Both native models appear in the opposite title code sets |
| 40 | `native_category_both_present` | Both native categories are present |
| 41 | `native_category_token_jaccard` | Jaccard overlap of category tokens |
| 42 | `native_category_exact` | Both normalized categories are present and equal |

The [feature code](../src/product_matching/features.py) and [title evidence code](../src/product_matching/lexical.py) define exact tokenization and arithmetic; the dictionary describes those operations without renaming the API.

## Model input and labels

A pair record contains `pair_id`, `left`, `right`, `left_native` and `right_native`. Only a `fit` record includes `label`, an integer 0 or 1. Development, calibration and evaluation labels are separate lists of `{pair_id, label}` records and are joined to scores by ID for their specific commands.

`project-role` joins supplied frozen views, native sidecars and ordered membership IDs. It does not create a split. `prepare` creates the 42 features and AB/BA token inputs; non-fit preparation requires the saved fit IDF state. [CORE_USAGE](CORE_USAGE.md) gives the file conventions and commands.

## Access and reconstruction

The checkout contains no original product rows, labels, membership lists, snapshots or trained weights. Acme examples in the Notebook and pages are synthetic illustrations, not observations or model predictions. Third-party data must be obtained and used under its own terms.

The frozen role inputs can be reconstructed from a locally obtained original ZIP using `reconstruct-roles` and the bundled source-position recipe. All four business files matched the retained inputs byte for byte. The command preserves original missing-field masks and separates non-fit labels. It does not choose a new split or recover historical exposure. Substituting a random split would produce a different experiment. See [reproduction](REPRODUCIBILITY.md) and [limitations](LIMITATIONS.md).
