# NLP

`TextParser` constructs a tokenizer for every `Locale`. Language-specific regex
splitters keep their existing behavior. Locales without one use a warned Unicode
word fallback: whitespace and punctuation separate words, letters and numbers
remain intact, and Latin text is lowercased. JP, KO and ZH use this fallback
without stemming. Indexing and queries must use the same locale; this fallback
does not provide dictionary segmentation of contiguous CJK text.
