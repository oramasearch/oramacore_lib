use super::stop_words::{StopWords, get_stop_words};
use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Locale {
    AR,
    BG,
    DA,
    DE,
    EL,
    EN,
    ES,
    ET,
    FI,
    FR,
    GA,
    HI,
    HU,
    HY,
    ID,
    IT,
    JP,
    KO,
    LT,
    NE,
    NL,
    NO,
    PT,
    RO,
    RU,
    SA,
    SL,
    SR,
    SV,
    TA,
    TR,
    UK,
    ZH,
}

// Compile-time guard for `Locale::ALL`. The exhaustive match stops compiling
// when a variant is added; add it to `ALL` too. The loop checks that `ALL`
// follows declaration order with no gaps or duplicates.
const _: () = {
    match Locale::AR {
        Locale::AR
        | Locale::BG
        | Locale::DA
        | Locale::DE
        | Locale::EL
        | Locale::EN
        | Locale::ES
        | Locale::ET
        | Locale::FI
        | Locale::FR
        | Locale::GA
        | Locale::HI
        | Locale::HU
        | Locale::HY
        | Locale::ID
        | Locale::IT
        | Locale::JP
        | Locale::KO
        | Locale::LT
        | Locale::NE
        | Locale::NL
        | Locale::NO
        | Locale::PT
        | Locale::RO
        | Locale::RU
        | Locale::SA
        | Locale::SL
        | Locale::SR
        | Locale::SV
        | Locale::TA
        | Locale::TR
        | Locale::UK
        | Locale::ZH => {}
    }
    assert!(Locale::ALL.len() == Locale::ZH as usize + 1);
    let mut i = 0;
    while i < Locale::ALL.len() {
        assert!(Locale::ALL[i] as usize == i);
        i += 1;
    }
};

impl Display for Locale {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            Locale::AR => write!(f, "arabic"),
            Locale::BG => write!(f, "bulgarian"),
            Locale::DA => write!(f, "danish"),
            Locale::DE => write!(f, "german"),
            Locale::EL => write!(f, "greek"),
            Locale::EN => write!(f, "english"),
            Locale::ET => write!(f, "estonian"),
            Locale::ES => write!(f, "spanish"),
            Locale::FI => write!(f, "finnish"),
            Locale::FR => write!(f, "french"),
            Locale::GA => write!(f, "irish"),
            Locale::HI => write!(f, "hindi"),
            Locale::HU => write!(f, "hungarian"),
            Locale::HY => write!(f, "armenian"),
            Locale::ID => write!(f, "indonesian"),
            Locale::IT => write!(f, "italian"),
            Locale::JP => write!(f, "japanese"),
            Locale::KO => write!(f, "korean"),
            Locale::LT => write!(f, "lithuanian"),
            Locale::NE => write!(f, "nepali"),
            Locale::NL => write!(f, "dutch"),
            Locale::NO => write!(f, "norwegian"),
            Locale::PT => write!(f, "portuguese"),
            Locale::RO => write!(f, "romanian"),
            Locale::RU => write!(f, "russian"),
            Locale::SA => write!(f, "sanskrit"),
            Locale::SL => write!(f, "slovenian"),
            Locale::SR => write!(f, "serbian"),
            Locale::SV => write!(f, "swedish"),
            Locale::TA => write!(f, "tamil"),
            Locale::TR => write!(f, "turkish"),
            Locale::UK => write!(f, "ukrainian"),
            Locale::ZH => write!(f, "chinese"),
        }
    }
}

impl FromStr for Locale {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Locale> {
        match s {
            "ar" => Ok(Locale::AR),
            "bg" => Ok(Locale::BG),
            "da" => Ok(Locale::DA),
            "de" => Ok(Locale::DE),
            "el" => Ok(Locale::EL),
            "en" => Ok(Locale::EN),
            "es" => Ok(Locale::ES),
            "et" => Ok(Locale::ET),
            "fi" => Ok(Locale::FI),
            "fr" => Ok(Locale::FR),
            "ga" => Ok(Locale::GA),
            "hi" => Ok(Locale::HI),
            "hu" => Ok(Locale::HU),
            "hy" => Ok(Locale::HY),
            "id" => Ok(Locale::ID),
            "it" => Ok(Locale::IT),
            "jp" => Ok(Locale::JP),
            "ko" => Ok(Locale::KO),
            "lt" => Ok(Locale::LT),
            "ne" => Ok(Locale::NE),
            "nl" => Ok(Locale::NL),
            "no" => Ok(Locale::NO),
            "pt" => Ok(Locale::PT),
            "ro" => Ok(Locale::RO),
            "ru" => Ok(Locale::RU),
            "sa" => Ok(Locale::SA),
            "sl" => Ok(Locale::SL),
            "sr" => Ok(Locale::SR),
            "sv" => Ok(Locale::SV),
            "ta" => Ok(Locale::TA),
            "tr" => Ok(Locale::TR),
            "uk" => Ok(Locale::UK),
            "zh" => Ok(Locale::ZH),

            "arabic" => Ok(Locale::AR),
            "bulgarian" => Ok(Locale::BG),
            "danish" => Ok(Locale::DA),
            "german" => Ok(Locale::DE),
            "estonian" => Ok(Locale::ET),
            "greek" => Ok(Locale::EL),
            "english" => Ok(Locale::EN),
            "spanish" => Ok(Locale::ES),
            "finnish" => Ok(Locale::FI),
            "french" => Ok(Locale::FR),
            "irish" => Ok(Locale::GA),
            "hindi" => Ok(Locale::HI),
            "hungarian" => Ok(Locale::HU),
            "armenian" => Ok(Locale::HY),
            "indonesian" => Ok(Locale::ID),
            "italian" => Ok(Locale::IT),
            "japanese" => Ok(Locale::JP),
            "korean" => Ok(Locale::KO),
            "lithuanian" => Ok(Locale::LT),
            "nepali" => Ok(Locale::NE),
            "dutch" => Ok(Locale::NL),
            "norwegian" => Ok(Locale::NO),
            "portuguese" => Ok(Locale::PT),
            "romanian" => Ok(Locale::RO),
            "russian" => Ok(Locale::RU),
            "sanskrit" => Ok(Locale::SA),
            "slovenian" => Ok(Locale::SL),
            "serbian" => Ok(Locale::SR),
            "swedish" => Ok(Locale::SV),
            "tamil" => Ok(Locale::TA),
            "turkish" => Ok(Locale::TR),
            "ukrainian" => Ok(Locale::UK),
            "chinese" => Ok(Locale::ZH),

            _ => anyhow::bail!("Invalid locale: {}", s),
        }
    }
}

impl Locale {
    /// Every variant, in declaration order.
    pub const ALL: [Locale; 33] = [
        Locale::AR,
        Locale::BG,
        Locale::DA,
        Locale::DE,
        Locale::EL,
        Locale::EN,
        Locale::ES,
        Locale::ET,
        Locale::FI,
        Locale::FR,
        Locale::GA,
        Locale::HI,
        Locale::HU,
        Locale::HY,
        Locale::ID,
        Locale::IT,
        Locale::JP,
        Locale::KO,
        Locale::LT,
        Locale::NE,
        Locale::NL,
        Locale::NO,
        Locale::PT,
        Locale::RO,
        Locale::RU,
        Locale::SA,
        Locale::SL,
        Locale::SR,
        Locale::SV,
        Locale::TA,
        Locale::TR,
        Locale::UK,
        Locale::ZH,
    ];

    pub fn to_iso_code_639_2(&self) -> &str {
        match self {
            Locale::AR => "ar",
            Locale::BG => "bg",
            Locale::DA => "da",
            Locale::DE => "de",
            Locale::EL => "el",
            Locale::EN => "en",
            Locale::ES => "es",
            Locale::ET => "et",
            Locale::FI => "fi",
            Locale::FR => "fr",
            Locale::GA => "ga",
            Locale::HI => "hi",
            Locale::HU => "hu",
            Locale::HY => "hy",
            Locale::ID => "id",
            Locale::IT => "it",
            Locale::JP => "jp",
            Locale::KO => "ko",
            Locale::LT => "lt",
            Locale::NE => "ne",
            Locale::NL => "nl",
            Locale::NO => "no",
            Locale::PT => "pt",
            Locale::RO => "ro",
            Locale::RU => "ru",
            Locale::SA => "sa",
            Locale::SL => "sl",
            Locale::SR => "sr",
            Locale::SV => "sv",
            Locale::TA => "ta",
            Locale::TR => "tr",
            Locale::UK => "uk",
            Locale::ZH => "zh",
        }
    }

    /// Regex matching the separators between words. JP, KO and ZH split on
    /// anything that is not a letter, mark or digit; the tokenizer then
    /// segments CJK runs into bigrams.
    pub fn split_regex(&self) -> Regex {
        match self {
            Locale::AR => Regex::new(r"(?im)[^a-z0-9أ-ي]+").unwrap(),
            Locale::BG => Regex::new(r"(?im)[^a-z0-9а-яА-Я]+").unwrap(),
            Locale::DA => Regex::new(r"(?im)[^a-z0-9æøåÆØÅ]+").unwrap(),
            Locale::DE => Regex::new(r"(?im)[^a-z0-9A-ZäöüÄÖÜß]+").unwrap(),
            Locale::EL => Regex::new(r"(?im)[^a-z0-9α-ωά-ώ]+").unwrap(),
            Locale::EN => Regex::new(r"(?im)[^A-Za-zàèéìòóù0-9_'\u{2019}]+").unwrap(),
            Locale::ES => Regex::new(r"(?im)[^a-z0-9A-Zá-úÁ-ÚñÑüÜ]+").unwrap(),
            Locale::ET => Regex::new(r"(?im)[^a-z0-9A-Zõäöüšž]+").unwrap(),
            Locale::FI => Regex::new(r"(?im)[^a-z0-9äöÄÖ]+").unwrap(),
            Locale::FR => Regex::new(r"(?im)[^a-z0-9äâàéèëêïîöôùüûœç-]+").unwrap(),
            Locale::GA => Regex::new(r"(?im)[^a-z0-9áéíóúÁÉÍÓÚ]+").unwrap(),
            Locale::HI => Regex::new(r"(?im)[^a-z0-9अ-आइ-ईउ-ऊऋ-एऐ-औअंअः]+").unwrap(),
            Locale::HU => Regex::new(r"(?im)[^a-z0-9áéíóöőúüűÁÉÍÓÖŐÚÜŰ]+").unwrap(),
            Locale::HY => Regex::new(r"(?im)[^a-z0-9ա-ֆ]+").unwrap(),
            Locale::ID => Regex::new(r"(?im)[^a-z0-9]+").unwrap(),
            Locale::IT => Regex::new(r"(?im)[^A-Za-zàèéìòóù0-9_'-]+").unwrap(),
            Locale::JP | Locale::KO | Locale::ZH => Regex::new(r"[^\p{L}\p{M}\p{N}]+").unwrap(),
            Locale::LT => Regex::new(r"(?im)[^a-z0-9ąčęėįšųūžĄČĘĖĮŠŲŪŽ]+").unwrap(),
            Locale::NE => Regex::new(r"(?im)[^a-z0-9अ-ह]+").unwrap(),
            Locale::NL => Regex::new(r"(?im)[^A-Za-zàèéìòóù0-9_'-]+").unwrap(),
            Locale::NO => Regex::new(r"(?im)[^a-z0-9_æøåÆØÅäÄöÖüÜ]+").unwrap(),
            Locale::PT => Regex::new(r"(?im)[^a-z0-9à-úÀ-Ú]+").unwrap(),
            Locale::RO => Regex::new(r"(?im)[^a-z0-9ăâîșțĂÂÎȘȚ]+").unwrap(),
            Locale::RU => Regex::new(r"(?im)[^a-z0-9а-яА-ЯёЁ]+").unwrap(),
            Locale::SA => Regex::new(r"(?im)[^a-z0-9A-Zāīūṛḷṃṁḥśṣṭḍṇṅñḻḹṝ]+").unwrap(),
            Locale::SL => Regex::new(r"(?im)[^a-z0-9čžšČŽŠ]+").unwrap(),
            Locale::SR => Regex::new(r"(?im)[^a-z0-9čćžšđČĆŽŠĐ]+").unwrap(),
            Locale::SV => Regex::new(r"(?im)[^a-z0-9_åÅäÄöÖüÜ-]+").unwrap(),
            Locale::TA => Regex::new(r"(?im)[^a-z0-9அ-ஹ]+").unwrap(),
            Locale::TR => Regex::new(r"(?im)[^a-z0-9çÇğĞıİöÖşŞüÜ]+").unwrap(),
            Locale::UK => Regex::new(r"(?im)[^a-z0-9а-яА-ЯіїєІЇЄ]+").unwrap(),
        }
    }

    pub fn stop_words(&self) -> Option<StopWords> {
        get_stop_words(*self)
    }
}
