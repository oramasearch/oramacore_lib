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
/// let result = parse("enum(\"engineer\")");
///
/// assert_eq!(result, Ok(Some("engineer".to_string()));
/// ```
use thiserror::Error;

#[derive(Error, Debug, PartialEq)]
pub enum EnumerativeParserError {
    #[error(
        "Parse error: Expected single ('), double (\") or backtick (`) quote at position {0}. Found \"{1}\" instead."
    )]
    InvalidQuoteChar(usize, char),

    #[error("Parse error: Expected closing parenthesis at the end of enum call.")]
    MissingCloseParenthesis,

    #[error(
        "Parse error: Mismatched closing quote for enum value. Expected \" {0} \" but found \" {1} \"."
    )]
    MismatchedClosingQuote(char, char),

    #[error("Parse error: Missing closing quote character due to escape")]
    MissingClosingCharacter,

    #[error("Missin enum value")]
    MissingEnumValue,
}

pub fn parse(value: &str) -> Result<Option<String>, EnumerativeParserError> {
    // Early exit in case it doesn't start with "enum(".
    if !value.starts_with("enum(") {
        return Ok(None);
    }

    // If the last character is not a closing parenthesis, it's an error.
    if !value.ends_with(")") {
        return Err(EnumerativeParserError::MissingCloseParenthesis);
    }

    // If the second to last character is not a matching quote, it's an error.
    let maybe_closing_quote: &str = &value[5..value.len() - 1];
    let mut maybe_closing_quote_chars = maybe_closing_quote.chars();

    let first_quote_char = maybe_closing_quote_chars.next();
    let last_quote_char = maybe_closing_quote_chars.last();

    if first_quote_char != last_quote_char {
        dbg!(&first_quote_char, &last_quote_char);

        return Err(EnumerativeParserError::MismatchedClosingQuote(
            maybe_closing_quote.chars().next().unwrap_or(' '),
            maybe_closing_quote.chars().last().unwrap_or(' '),
        ));
    }

    let quote_char = if let Some(first_char) = first_quote_char {
        static ALLOWED_QUOTE_CHARS: [char; 3] = ['"', '\'', '`'];

        if !ALLOWED_QUOTE_CHARS.contains(&first_char) {
            return Err(EnumerativeParserError::InvalidQuoteChar(0, first_char));
        }

        first_char
    } else {
        // case: "enum()"
        return Err(EnumerativeParserError::MissingEnumValue);
    };

    // enum("...
    // so it's time to start reading and accumulating the enum value.
    let escaped_quote = &maybe_closing_quote[1..maybe_closing_quote.len() - 1];
    let mut enum_value = String::with_capacity(escaped_quote.len());
    let mut characters = escaped_quote.chars();

    loop {
        let Some(current_char) = characters.next() else {
            break;
        };

        if current_char == '\\' {
            let Some(next_char) = characters.next() else {
                return Err(EnumerativeParserError::MissingClosingCharacter);
            };

            if next_char == quote_char {
                enum_value.push(next_char);
            } else {
                enum_value.push(current_char);
                enum_value.push(next_char);
            }
        } else {
            enum_value.push(current_char);
        }
    }

    Ok(Some(enum_value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_enum_parsing() {
        let result = parse("enum(\"engineer\")");
        assert_eq!(result, Ok(Some("engineer".to_string())));
    }

    #[test]
    fn simple_enum_parsing_single_quotes() {
        let result = parse("enum('engineer')");
        assert_eq!(result, Ok(Some("engineer".to_string())));
    }

    #[test]
    fn simple_enum_parsing_backtick_quotes() {
        let result = parse("enum(`engineer`)");
        assert_eq!(result, Ok(Some("engineer".to_string())));
    }

    #[test]
    fn enum_parsing_with_escaped_quotes() {
        let result = parse("enum(\"engi\\\"neer\")");

        assert_eq!(result, Ok(Some("engi\"neer".to_string())));
    }

    #[test]
    fn enum_parsing_with_mixed_quotes() {
        let result = parse("enum('engi`neer')");

        assert_eq!(result, Ok(Some("engi`neer".to_string())));
    }

    #[test]
    fn enum_parsing_with_escaped_backslash() {
        let result = parse("enum(\"engi\\neer\")");
        assert_eq!(result, Ok(Some("engi\\neer".to_string())));
    }

    // Remember: if it doesn't start with enum(, it's not an enum.
    // In that case, we do not expect an error.
    #[test]
    fn enum_parsing_missing_opening_parenthesis() {
        let result = parse("enum\"engineer\")");
        assert_eq!(result, Ok(None));
    }

    #[test]
    fn enum_parsing_missing_opening_quote() {
        let result = parse("enum(engineer\")");
        assert!(result.is_err());
    }

    #[test]
    fn enum_parsing_missing_closing_parenthesis() {
        let result = parse("enum(\"engineer\"");
        assert!(result.is_err());
    }

    #[test]
    fn enum_parsing_missing_closing_quote() {
        let result = parse("enum(\"engineer)");
        assert!(result.is_err());
    }

    #[test]
    fn enum_parsing_mismatched_closing_quote() {
        let result = parse("enum(\"engineer')");
        assert!(result.is_err());
    }
}
