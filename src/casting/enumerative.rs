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
/// let mut parser = Enumerative::new("enum(\"engineer\")");
///
/// let result = parser.parse();
/// assert!(result.is_enum);
/// assert_eq!(result.enum_value, Some("engineer".to_string()));
/// ```

static ESCAPABLE_CHARS: [char; 4] = ['\'', '"', '`', ')'];

#[derive(Clone, Debug, PartialEq)]
enum QuoteStyle {
    Single,
    Double,
    Backtick,
}

#[derive(Clone, Debug)]
pub struct Enumerative {
    pub is_enum: bool,
    pub enum_value: Option<String>,

    raw_value: String,
    chars: Vec<char>,
    cursor: usize,
    inside_enum: bool,
    quote_char: Option<QuoteStyle>,
}

impl Enumerative {
    pub fn new(value: &str) -> Self {
        Enumerative {
            is_enum: false,
            enum_value: None,
            raw_value: value.to_string().trim().to_string(),
            chars: value.chars().collect(),
            cursor: 0,
            inside_enum: false,
            quote_char: None,
        }
    }

    pub fn parse(&mut self) -> Result<Self, Box<dyn std::error::Error>> {
        // Early exit in case it doesn't start with "enum(".
        if !self.raw_value.starts_with("enum(") {
            return Ok(self.clone());
        }

        self.inside_enum = true;
        self.cursor += 5; // Move past "enum("
        self.quote_char = self.peek_quote_char_at(None);

        // The first character of an enum call should be either a single, double, or backtick quote.
        if self.quote_char.is_none() {
            let err = format!(
                "Parse error: Expected single ('), double (\") or backtick (`) quote at position {}. Found \"{}\" instead.",
                self.cursor,
                self.chars.get(self.cursor).unwrap_or(&' ')
            );
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                err,
            )));
        }

        // If the last character is not a closing parenthesis, it's an error.
        if self.chars.last() != Some(&')') {
            let err = format!(
                "Parse error: Expected closing parenthesis at the end of enum call. Found \"{}\" instead.",
                self.chars.last().unwrap_or(&' ')
            );
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                err,
            )));
        }

        // If the second to last character is not a matching quote, it's an error.
        let last_but_one = self.chars.len().checked_sub(2);
        if self.peek_quote_char_at(last_but_one) != self.quote_char {
            let err = format!(
                "Parse error: Mismatched closing quote for enum value. Expected {:?} but found {:?}.",
                self.quote_char,
                self.peek_quote_char_at(last_but_one)
            );
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                err,
            )));
        }

        // If no error is found, we can safely assume we're inside an enum call.
        self.inside_enum = true;
        // Move past the opening quote.
        self.cursor += 1;

        // Cursor now should be in position 6:
        // enum("...
        // so it's time to start reading and accumulating the enum value.
        let mut enum_value = String::new();

        while self.cursor < self.chars.len() {
            let current_char = self.chars[self.cursor];
            dbg!("Current char: {}", current_char);

            // Current char could be an escaped quote, parenthesis, or backlash.
            if current_char == '\\' {
                let next_char = self.chars.get(self.cursor + 1);
                if ESCAPABLE_CHARS.contains(next_char.unwrap_or(&' ')) {
                    enum_value.push(*next_char.unwrap());
                    self.cursor += 2; // Skip the escape character and the escaped character.
                    continue;
                } else {
                    enum_value.push(current_char);
                    self.cursor += 1;
                    continue;
                }
            }

            // If we find the closing quote, we should stop accumulating.
            if let Some(quote_style) = &self.quote_char {
                let is_closing_quote = match quote_style {
                    QuoteStyle::Single => current_char == '\'',
                    QuoteStyle::Double => current_char == '"',
                    QuoteStyle::Backtick => current_char == '`',
                };
                if is_closing_quote {
                    break;
                }
            }

            self.cursor += 1;
            enum_value.push(current_char);
        }

        Ok(Enumerative {
            is_enum: true,
            enum_value: Some(enum_value),
            ..self.clone()
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
        let mut parser = Enumerative::new("enum(\"engineer\")");

        let result = parser.parse().unwrap();
        assert!(result.is_enum);
        assert_eq!(result.enum_value, Some("engineer".to_string()));
    }

    #[test]
    fn simple_enum_parsing_single_quotes() {
        let mut parser = Enumerative::new("enum('engineer')");
        let result = parser.parse().unwrap();
        assert!(result.is_enum);
        assert_eq!(result.enum_value, Some("engineer".to_string()));
    }

    #[test]
    fn simple_enum_parsing_backtick_quotes() {
        let mut parser = Enumerative::new("enum(`engineer`)");
        let result = parser.parse().unwrap();
        assert!(result.is_enum);
        assert_eq!(result.enum_value, Some("engineer".to_string()));
    }

    #[test]
    fn enum_parsing_with_escaped_quotes() {
        let mut parser = Enumerative::new("enum(\"engi\\\"neer\")");
        let result = parser.parse().unwrap();
        assert!(result.is_enum);
        assert_eq!(result.enum_value, Some("engi\"neer".to_string()));
    }

    #[test]
    fn enum_parsing_with_mixed_quotes() {
        let mut parser = Enumerative::new("enum('engi`neer')");
        let result = parser.parse().unwrap();
        assert!(result.is_enum);
        assert_eq!(result.enum_value, Some("engi`neer".to_string()));
    }

    #[test]
    fn enum_parsing_with_escaped_backslash() {
        let mut parser = Enumerative::new("enum(\"engi\\neer\")");
        let result = parser.parse().unwrap();
        assert!(result.is_enum);
        assert_eq!(result.enum_value, Some("engi\\neer".to_string()));
    }

    // Remember: if it doesn't start with enum(, it's not an enum.
    // In that case, we do not expect an error.
    #[test]
    fn enum_parsing_missing_opening_parenthesis() {
        let mut parser = Enumerative::new("enum\"engineer\")");
        let result = parser.parse().unwrap();
        assert!(!result.is_enum);
        assert_eq!(result.enum_value, None);
    }

    #[test]
    fn enum_parsing_missing_opening_quote() {
        let mut parser = Enumerative::new("enum(engineer\")");
        let result = parser.parse();
        assert!(result.is_err());
    }

    #[test]
    fn enum_parsing_missing_closing_parenthesis() {
        let mut parser = Enumerative::new("enum(\"engineer\"");
        let result = parser.parse();
        assert!(result.is_err());
    }

    #[test]
    fn enum_parsing_missing_closing_quote() {
        let mut parser = Enumerative::new("enum(\"engineer)");
        let result = parser.parse();
        assert!(result.is_err());
    }

    #[test]
    fn enum_parsing_mismatched_closing_quote() {
        let mut parser = Enumerative::new("enum(\"engineer')");
        let result = parser.parse();
        assert!(result.is_err());
    }
}
