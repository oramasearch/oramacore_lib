use std::collections::HashSet;

use super::locales::Locale;
use regex::Regex;

#[derive(Debug, Clone)]
pub struct Tokenizer {
    split_regex: Regex,
    stop_words: HashSet<&'static str>,
    /// Segment CJK runs into bigrams (JP, KO, ZH). See [`Tokenizer::tokenize`].
    cjk_bigrams: bool,
}

/// List of tokens that should never be treated as stop words,
/// even if they are present in the stop word list of a locale.
static NEVER_TOKENS: &[&str] = &["ai", "AI"];

impl Tokenizer {
    fn new(locale: Locale) -> Self {
        let mut stop_words: HashSet<&'static str> = locale.stop_words().unwrap_or_default();
        for &token in NEVER_TOKENS {
            stop_words.remove(token);
        }

        Tokenizer {
            split_regex: locale.split_regex(),
            stop_words,
            cjk_bigrams: matches!(locale, Locale::JP | Locale::KO | Locale::ZH),
        }
    }

    pub fn arab() -> Self {
        Self::new(Locale::AR)
    }

    pub fn bulgarian() -> Self {
        Self::new(Locale::BG)
    }

    pub fn danish() -> Self {
        Self::new(Locale::DA)
    }

    pub fn german() -> Self {
        Self::new(Locale::DE)
    }

    pub fn greek() -> Self {
        Self::new(Locale::EL)
    }

    pub fn english() -> Self {
        Self::new(Locale::EN)
    }

    pub fn spanish() -> Self {
        Self::new(Locale::ES)
    }

    pub fn estonian() -> Self {
        Self::new(Locale::ET)
    }

    pub fn finnish() -> Self {
        Self::new(Locale::FI)
    }

    pub fn french() -> Self {
        Self::new(Locale::FR)
    }

    pub fn irish() -> Self {
        Self::new(Locale::GA)
    }

    pub fn hindi() -> Self {
        Self::new(Locale::HI)
    }

    pub fn hungarian() -> Self {
        Self::new(Locale::HU)
    }

    pub fn armenian() -> Self {
        Self::new(Locale::HY)
    }

    pub fn indonesian() -> Self {
        Self::new(Locale::ID)
    }

    pub fn italian() -> Self {
        Self::new(Locale::IT)
    }

    pub fn japanese() -> Self {
        Self::new(Locale::JP)
    }

    pub fn korean() -> Self {
        Self::new(Locale::KO)
    }

    pub fn lithuanian() -> Self {
        Self::new(Locale::LT)
    }

    pub fn nepali() -> Self {
        Self::new(Locale::NE)
    }

    pub fn dutch() -> Self {
        Self::new(Locale::NL)
    }

    pub fn norwegian() -> Self {
        Self::new(Locale::NO)
    }

    pub fn portuguese() -> Self {
        Self::new(Locale::PT)
    }

    pub fn romanian() -> Self {
        Self::new(Locale::RO)
    }

    pub fn russian() -> Self {
        Self::new(Locale::RU)
    }

    pub fn sanskrit() -> Self {
        Self::new(Locale::SA)
    }

    pub fn slovenian() -> Self {
        Self::new(Locale::SL)
    }

    pub fn serbian() -> Self {
        Self::new(Locale::SR)
    }

    pub fn swedish() -> Self {
        Self::new(Locale::SV)
    }

    pub fn tamil() -> Self {
        Self::new(Locale::TA)
    }

    pub fn turkish() -> Self {
        Self::new(Locale::TR)
    }

    pub fn ukrainian() -> Self {
        Self::new(Locale::UK)
    }

    pub fn chinese() -> Self {
        Self::new(Locale::ZH)
    }

    /// Splits `input` into lowercase, stop-word-filtered tokens.
    ///
    /// For JP, KO and ZH, which do not separate words with spaces, each word is
    /// further split where the script changes between CJK (Han, Hiragana,
    /// Katakana, Hangul) and anything else. A CJK run becomes overlapping
    /// character bigrams (`東京都` → `東京`, `京都`), or the single character when
    /// the run has length 1; other runs (Latin words, digits) stay whole.
    pub fn tokenize<'a, 'b>(&'a self, input: &'b str) -> impl Iterator<Item = String> + 'b
    where
        'a: 'b,
    {
        self.split_regex
            .split(input)
            .filter(|token| !token.is_empty())
            .flat_map(|word| segment(word, self.cjk_bigrams))
            .filter_map(|token| self.normalize_token(token.to_lowercase()))
            .filter(|token| !token.is_empty() && !self.stop_words.contains(token.as_str()))
    }

    fn normalize_token(&self, token: String) -> Option<String> {
        if self.stop_words.contains(token.as_str()) {
            return None;
        }
        Some(self.replace_diacritics(token.as_str()))
    }

    fn replace_diacritics(&self, str: &str) -> String {
        str.chars().map(replace_char).collect()
    }
}

/// Yields `word` whole, or, when `cjk_bigrams` is set, its non-CJK runs whole
/// and its CJK runs as overlapping bigrams.
fn segment(word: &str, cjk_bigrams: bool) -> impl Iterator<Item = &str> {
    let bigram_char = move |c: char| cjk_bigrams && is_cjk(c);
    let mut rest = word;
    // Set once a bigram has covered the first character of `rest`.
    let mut covered = false;

    std::iter::from_fn(move || {
        loop {
            let mut chars = rest.chars();
            let first = chars.next()?;
            if !bigram_char(first) {
                let end = rest.find(bigram_char).unwrap_or(rest.len());
                let (run, tail) = rest.split_at(end);
                rest = tail;
                covered = false;
                return Some(run);
            }

            let first_len = first.len_utf8();
            match chars.next().filter(|&c| bigram_char(c)) {
                Some(second) => {
                    let bigram = &rest[..first_len + second.len_utf8()];
                    rest = &rest[first_len..];
                    covered = true;
                    return Some(bigram);
                }
                None => {
                    let single = &rest[..first_len];
                    rest = &rest[first_len..];
                    if !std::mem::take(&mut covered) {
                        return Some(single);
                    }
                }
            }
        }
    })
}

/// Han, Hiragana, Katakana and Hangul letters, including half-width forms.
fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{1100}'..='\u{11FF}'     // Hangul Jamo
        | '\u{3005}' | '\u{3007}'   // Ideographic iteration mark, number zero
        | '\u{3040}'..='\u{30FF}'   // Hiragana, Katakana
        | '\u{3130}'..='\u{318F}'   // Hangul Compatibility Jamo
        | '\u{31F0}'..='\u{31FF}'   // Katakana Phonetic Extensions
        | '\u{3400}'..='\u{4DBF}'   // CJK Unified Ideographs Extension A
        | '\u{4E00}'..='\u{9FFF}'   // CJK Unified Ideographs
        | '\u{A960}'..='\u{A97F}'   // Hangul Jamo Extended-A
        | '\u{AC00}'..='\u{D7FF}'   // Hangul Syllables, Hangul Jamo Extended-B
        | '\u{F900}'..='\u{FAFF}'   // CJK Compatibility Ideographs
        | '\u{FF66}'..='\u{FFDC}'   // Half-width Katakana and Hangul
        | '\u{20000}'..='\u{323AF}' // CJK Extensions B to H, Compatibility Supplement
    )
}

fn replace_char(c: char) -> char {
    if c == '\u{2019}' {
        return '\'';
    }
    let code = u32::from(c);
    if !(DIACRITICS_CHARCODE_START..=DIACRITICS_CHARCODE_END).contains(&code) {
        return c;
    }

    let index = usize::try_from(code - DIACRITICS_CHARCODE_START)
        .expect("Invalid index: code - DIACRITICS_CHARCODE_START is not a valid usize");
    *CHARCODE_REPLACE_MAPPING.get(index).unwrap_or(&c)
}

const CHARCODE_REPLACE_MAPPING: &[char] = &[
    'A', 'A', 'A', 'A', 'A', 'A', 'A', 'C', 'E', 'E', 'E', 'E', 'I', 'I', 'I', 'I', 'E', 'N', 'O',
    'O', 'O', 'O', 'O', '\0', 'O', 'U', 'U', 'U', 'U', 'Y', 'P', 's', 'a', 'a', 'a', 'a', 'a', 'a',
    'a', 'c', 'e', 'e', 'e', 'e', 'i', 'i', 'i', 'i', 'e', 'n', 'o', 'o', 'o', 'o', 'o', '\0', 'o',
    'u', 'u', 'u', 'u', 'y', 'p', 'y', 'A', 'a', 'A', 'a', 'A', 'a', 'C', 'c', 'C', 'c', 'C', 'c',
    'C', 'c', 'D', 'd', 'D', 'd', 'E', 'e', 'E', 'e', 'E', 'e', 'E', 'e', 'E', 'e', 'G', 'g', 'G',
    'g', 'G', 'g', 'G', 'g', 'H', 'h', 'H', 'h', 'I', 'i', 'I', 'i', 'I', 'i', 'I', 'i', 'I', 'i',
    'I', 'i', 'J', 'j', 'K', 'k', 'k', 'L', 'l', 'L', 'l', 'L', 'l', 'L', 'l', 'L', 'l', 'N', 'n',
    'N', 'n', 'N', 'n', 'n', 'N', 'n', 'O', 'o', 'O', 'o', 'O', 'o', 'O', 'o', 'R', 'r', 'R', 'r',
    'R', 'r', 'S', 's', 'S', 's', 'S', 's', 'S', 's', 'T', 't', 'T', 't', 'T', 't', 'U', 'u', 'U',
    'u', 'U', 'u', 'U', 'u', 'U', 'u', 'U', 'u', 'W', 'w', 'Y', 'y', 'Y', 'Z', 'z', 'Z', 'z', 'Z',
    'z', 's',
];

const DIACRITICS_CHARCODE_START: u32 = 192;
const DIACRITICS_CHARCODE_END: u32 = 383;

#[cfg(test)]
mod tests {
    use super::Locale;

    #[test]
    fn test_tokenizer() {
        let tokenizer = super::Tokenizer::english();
        let tokens: Vec<String> = tokenizer.tokenize("Hello, world!").collect();
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_tokenizer_2() {
        let tokenizer = super::Tokenizer::english();
        let tokens: Vec<String> = tokenizer.tokenize("Hello, - world!").collect();
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_tokenizer_unicode_apostrophe() {
        let tokenizer = super::Tokenizer::english();
        let tokens: Vec<String> = tokenizer.tokenize("five o\u{2019}clock train").collect();
        assert_eq!(tokens, vec!["five", "o'clock", "train"]);
    }

    #[test]
    fn test_tokenizer_hyphenated_words() {
        let tokenizer = super::Tokenizer::english();
        let tokens: Vec<String> = tokenizer.tokenize("well-known fact").collect();
        assert_eq!(tokens, vec!["well", "known", "fact"]);
    }

    #[test]
    fn test_tokenizer_ai_italian() {
        let tokenizer = super::Tokenizer::italian();
        let tokens: Vec<String> = tokenizer.tokenize("AI & ChatGPT").collect();
        assert_eq!(tokens, vec!["ai", "chatgpt"]);
    }

    fn tokens(locale: Locale, input: &str) -> Vec<String> {
        super::Tokenizer::new(locale).tokenize(input).collect()
    }

    #[test]
    fn test_tokenizer_builds_for_every_locale() {
        for locale in Locale::ALL {
            let tokenizer = super::Tokenizer::new(locale);
            let _: Vec<String> = tokenizer.tokenize("Hello 東京 2024").collect();
        }
    }

    #[test]
    fn test_tokenizer_japanese_bigrams() {
        assert_eq!(
            tokens(Locale::JP, "東京タワー"),
            vec!["東京", "京タ", "タワ", "ワー"]
        );
    }

    #[test]
    fn test_tokenizer_korean_bigrams() {
        assert_eq!(
            tokens(Locale::KO, "한국어 검색"),
            vec!["한국", "국어", "검색"]
        );
    }

    #[test]
    fn test_tokenizer_chinese_bigrams() {
        assert_eq!(
            tokens(Locale::ZH, "中文搜索引擎。"),
            vec!["中文", "文搜", "搜索", "索引", "引擎"]
        );
    }

    #[test]
    fn test_tokenizer_cjk_mixed_text() {
        assert_eq!(
            tokens(Locale::JP, "東京 Tower 2024"),
            vec!["東京", "tower", "2024"]
        );
        assert_eq!(
            tokens(Locale::ZH, "北京Tower2024号"),
            vec!["北京", "tower2024", "号"]
        );
    }

    #[test]
    fn test_tokenizer_cjk_single_character_run() {
        assert_eq!(tokens(Locale::ZH, "猫 cat"), vec!["猫", "cat"]);
    }
}
