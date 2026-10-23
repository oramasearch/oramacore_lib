# OramaCore internal library

This repository contains some internal structures and utilities used by OramaCore. Don't use it directly unless you know what you're doing.

##  License

See [LICENSE](./LICENSE) file for details.

## Locale tokenization

Every `Locale` variant can create a `TextParser`. Japanese (`JP`), Korean (`KO`),
and Chinese (`ZH`) use a Unicode word fallback and log a warning when a parser
is created. It splits on whitespace and punctuation, preserves letters and numbers,
and lowercases Latin text. It does not perform dictionary word segmentation or
stemming for these locales; queries must match the same contiguous words as indexing.
