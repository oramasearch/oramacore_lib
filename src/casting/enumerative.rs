/// Definition
/// An OramaCore enum is a value that can be one of several variants.
/// Unlike some languages, OramaCore enums do not support associated data with their variants.
///
/// While OramaCore considers enum all strings <=25 ASCII characters, you can enable `explicit_enums_only` to enforce
/// strict enum definition at insertion time. This way, only values wrapped in an `enum()` call will be considered enums.
///
/// Example:
/// ```json
/// {
///   "name": "Michele",
///   "job": "enum(\"engineer\")"
/// }
/// ```
///
/// With `explicit_enums_only` disabled, `"Michele"` would have been considered an enum, but with `explicit_enums_only` enabled,
/// it is treated as a string. Only `"engineer"` is treated as an enum.
///
/// Enumerative parser usage:
///
/// ```rs
/// let result = Enumerative::parse("enum(\"engineer\")");
/// assert!(result.is_enum);
/// assert_eq!(result.enum_value, Some("engineer".to_string()));
/// ```
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EnumerativeParserError {
    #[error(
        "Parse error: Expected single ('), double (\") or backtick (`) quote at position {0}. Found \"{1}\" instead."
    )]
    InvalidQuoteChar(usize, char),
    #[error(
        "Parse error: Expected closing parenthesis at the end of enum call. Found \"{0}\" instead."
    )]
    MissingCloseParenthesis(char),
    #[error(
        "Parse error: Mismatched closing quote for enum value. Expected \" {0} \" but found \" {1} \"."
    )]
    MismatchedClosingQuote(char, char),
}

static ESCAPABLE_CHARS: [char; 4] = ['\'', '"', '`', ')'];

#[derive(Clone, Debug, PartialEq)]
enum QuoteStyle {
    Single,
    Double,
    Backtick,
}

impl Into<char> for QuoteStyle {
    fn into(self) -> char {
        match self {
            QuoteStyle::Single => '\'',
            QuoteStyle::Double => '"',
            QuoteStyle::Backtick => '`',
        }
    }
}

#[derive(Clone, Debug)]
pub struct Enumerative<'input> {
    pub is_enum: bool,
    pub enum_value: Option<String>,

    raw_value: &'input str,
    chars: Vec<char>,
    cursor: usize,
    inside_enum: bool,
    quote_char: Option<QuoteStyle>,
}

impl<'input> Enumerative<'input> {
    pub fn parse(value: &'input str) -> Result<Self, EnumerativeParserError> {
        let mut parser = Enumerative {
            is_enum: false,
            enum_value: None,
            raw_value: value.trim(),
            chars: value.chars().collect(),
            cursor: 0,
            inside_enum: false,
            quote_char: None,
        };

        // Early exit in case it doesn't start with "enum(".
        if !parser.raw_value.starts_with("enum(") {
            return Ok(parser.clone());
        }

        parser.inside_enum = true;
        parser.cursor += 5; // Move past "enum("
        parser.quote_char = parser.peek_quote_char_at(None);

        // The first character of an enum call should be either a single, double, or backtick quote.
        if parser.quote_char.is_none() {
            return Err(EnumerativeParserError::InvalidQuoteChar(
                parser.cursor,
                *parser.chars.get(parser.cursor).unwrap_or(&' '),
            ));
        }

        // If the last character is not a closing parenthesis, it's an error.
        if parser.chars.last() != Some(&')') {
            return Err(EnumerativeParserError::MissingCloseParenthesis(
                *parser.chars.last().unwrap_or(&' '),
            ));
        }

        // If the second to last character is not a matching quote, it's an error.
        let last_but_one = parser.chars.len().checked_sub(2);
        if parser.peek_quote_char_at(last_but_one) != parser.quote_char {
            return Err(EnumerativeParserError::MismatchedClosingQuote(
                parser
                    .quote_char
                    .clone()
                    .unwrap_or(QuoteStyle::Double)
                    .into(), // @todo: we should never be in the 'or' condition of this unwrap.
                parser
                    .peek_quote_char_at(last_but_one)
                    .unwrap_or(QuoteStyle::Double)
                    .into(),
            ));
        }

        // If no error is found, we can safely assume we're inside an enum call.
        parser.inside_enum = true;
        // Move past the opening quote.
        parser.cursor += 1;

        // Cursor now should be in position 6:
        // enum("...
        // so it's time to start reading and accumulating the enum value.
        let mut enum_value = String::new();

        while parser.cursor < parser.chars.len() {
            let current_char = parser.chars[parser.cursor];
            dbg!("Current char: {}", current_char);

            // Current char could be an escaped quote, parenthesis, or backlash.
            if current_char == '\\' {
                let next_char = parser.chars.get(parser.cursor + 1);
                if ESCAPABLE_CHARS.contains(next_char.unwrap_or(&' ')) {
                    enum_value.push(*next_char.unwrap());
                    parser.cursor += 2; // Skip the escape character and the escaped character.
                    continue;
                } else {
                    enum_value.push(current_char);
                    parser.cursor += 1;
                    continue;
                }
            }

            // If we find the closing quote, we should stop accumulating.
            if let Some(quote_style) = &parser.quote_char {
                let is_closing_quote = match quote_style {
                    QuoteStyle::Single => current_char == '\'',
                    QuoteStyle::Double => current_char == '"',
                    QuoteStyle::Backtick => current_char == '`',
                };
                if is_closing_quote {
                    break;
                }
            }

            parser.cursor += 1;
            enum_value.push(current_char);
        }

        Ok(Enumerative {
            is_enum: true,
            enum_value: Some(enum_value),
            ..parser.clone()
        })
    }

    #[inline]
    fn peek_quote_char_at(&self, position: Option<usize>) -> Option<QuoteStyle> {
        let idx = position.unwrap_or(self.cursor);

        match self.chars.get(idx) {
            Some('"') => Some(QuoteStyle::Double),
            Some('\'') => Some(QuoteStyle::Single),
            Some('`') => Some(QuoteStyle::Backtick),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_enum_parsing() {
        let result = Enumerative::parse("enum(\"engineer\")").unwrap();
        assert!(result.is_enum);
        assert_eq!(result.enum_value, Some("engineer".to_string()));
    }

    #[test]
    fn simple_enum_parsing_single_quotes() {
        let result = Enumerative::parse("enum('engineer')").unwrap();
        assert!(result.is_enum);
        assert_eq!(result.enum_value, Some("engineer".to_string()));
    }

    #[test]
    fn simple_enum_parsing_backtick_quotes() {
        let result = Enumerative::parse("enum(`engineer`)").unwrap();
        assert!(result.is_enum);
        assert_eq!(result.enum_value, Some("engineer".to_string()));
    }

    #[test]
    fn enum_parsing_with_escaped_quotes() {
        let result = Enumerative::parse("enum(\"engi\\\"neer\")").unwrap();
        assert!(result.is_enum);
        assert_eq!(result.enum_value, Some("engi\"neer".to_string()));
    }

    #[test]
    fn enum_parsing_with_mixed_quotes() {
        let result = Enumerative::parse("enum('engi`neer')").unwrap();
        assert!(result.is_enum);
        assert_eq!(result.enum_value, Some("engi`neer".to_string()));
    }

    #[test]
    fn enum_parsing_with_escaped_backslash() {
        let result = Enumerative::parse("enum(\"engi\\neer\")").unwrap();
        assert!(result.is_enum);
        assert_eq!(result.enum_value, Some("engi\\neer".to_string()));
    }

    // Remember: if it doesn't start with enum(, it's not an enum.
    // In that case, we do not expect an error.
    #[test]
    fn enum_parsing_missing_opening_parenthesis() {
        let result = Enumerative::parse("enum\"engineer\")").unwrap();
        assert!(!result.is_enum);
        assert_eq!(result.enum_value, None);
    }

    #[test]
    fn enum_parsing_missing_opening_quote() {
        let result = Enumerative::parse("enum(engineer\")");
        assert!(result.is_err());
    }

    #[test]
    fn enum_parsing_missing_closing_parenthesis() {
        let result = Enumerative::parse("enum(\"engineer\"");
        assert!(result.is_err());
    }

    #[test]
    fn enum_parsing_missing_closing_quote() {
        let result = Enumerative::parse("enum(\"engineer)");
        assert!(result.is_err());
    }

    #[test]
    fn enum_parsing_mismatched_closing_quote() {
        let result = Enumerative::parse("enum(\"engineer')");
        assert!(result.is_err());
    }
}
