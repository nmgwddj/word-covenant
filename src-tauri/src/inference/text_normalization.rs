//! Deterministic local transcript projection without changing durable evidence.

use ferrous_opencc::{config::BuiltinConfig, OpenCC};
use std::sync::OnceLock;

pub const SIMPLIFIED_CHINESE_PROFILE: &str = "opencc-t2s/ferrous-opencc-0.4.0";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectedTranscriptText {
    pub text: String,
    pub original_text: Option<String>,
    pub normalization_profile: Option<String>,
}

pub fn project_simplified_chinese(raw: &str) -> ProjectedTranscriptText {
    static CONVERTER: OnceLock<OpenCC> = OnceLock::new();
    let converter = CONVERTER.get_or_init(|| {
        OpenCC::from_config(BuiltinConfig::T2s)
            .expect("embedded OpenCC t2s dictionaries must initialize")
    });
    let text = converter.convert(raw);
    let changed = text != raw;
    ProjectedTranscriptText {
        text,
        original_text: changed.then(|| raw.to_owned()),
        normalization_profile: changed.then(|| SIMPLIFIED_CHINESE_PROFILE.to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_traditional_chinese_without_changing_english_numbers_or_punctuation() {
        let projected = project_simplified_chinese("今天開會 review API v2.0，確認資料庫。");

        assert_eq!(projected.text, "今天开会 review API v2.0，确认资料库。");
        assert_eq!(
            projected.original_text.as_deref(),
            Some("今天開會 review API v2.0，確認資料庫。")
        );
        assert_eq!(
            projected.normalization_profile.as_deref(),
            Some(SIMPLIFIED_CHINESE_PROFILE)
        );
    }

    #[test]
    fn leaves_existing_simplified_text_unannotated_and_is_idempotent() {
        let first = project_simplified_chinese("今天开会 review API v2.0，确认资料库。");
        let second = project_simplified_chinese(&first.text);

        assert_eq!(first.text, second.text);
        assert_eq!(first.original_text, None);
        assert_eq!(first.normalization_profile, None);
        assert_eq!(second.original_text, None);
    }
}
