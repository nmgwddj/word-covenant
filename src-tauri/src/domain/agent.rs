use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;
use uuid::Uuid;

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DataCategory {
    SessionMetadata,
    Transcript,
    Summary,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolKind {
    LocalSpeech,
    Notification,
    HttpProfile,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ToolCall {
    LocalSpeech {
        text: String,
    },
    Notification {
        title: String,
        body: String,
    },
    HttpProfile {
        profile_id: String,
        origin: String,
        body: Value,
        data_categories: BTreeSet<DataCategory>,
    },
}

impl ToolCall {
    pub fn kind(&self) -> ToolKind {
        match self {
            Self::LocalSpeech { .. } => ToolKind::LocalSpeech,
            Self::Notification { .. } => ToolKind::Notification,
            Self::HttpProfile { .. } => ToolKind::HttpProfile,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::LocalSpeech { text } if text.trim().is_empty() => {
                Err("local speech requires non-empty text".to_owned())
            }
            Self::Notification { title, body }
                if title.trim().is_empty() || body.trim().is_empty() =>
            {
                Err("notification requires title and body".to_owned())
            }
            Self::HttpProfile {
                profile_id, origin, ..
            } if profile_id.trim().is_empty() || origin.trim().is_empty() => {
                Err("HTTP profile requires a profile identifier and origin".to_owned())
            }
            _ => Ok(()),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ActionProposal {
    pub id: Uuid,
    pub title: String,
    pub tool: ToolCall,
    pub requested_at: DateTime<Utc>,
    pub context_span_ids: Vec<Uuid>,
}

impl ActionProposal {
    pub fn new(
        title: impl Into<String>,
        tool: ToolCall,
        context_span_ids: Vec<Uuid>,
        requested_at: DateTime<Utc>,
    ) -> Result<Self, String> {
        let title = title.into();
        if title.trim().is_empty() {
            return Err("action title must not be empty".to_owned());
        }
        tool.validate()?;

        Ok(Self {
            id: Uuid::new_v4(),
            title,
            tool,
            requested_at,
            context_span_ids,
        })
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanV1 {
    pub id: Uuid,
    pub summary: String,
    pub actions: Vec<ActionProposal>,
    pub created_at: DateTime<Utc>,
}

impl PlanV1 {
    pub fn new(
        summary: impl Into<String>,
        actions: Vec<ActionProposal>,
        created_at: DateTime<Utc>,
    ) -> Result<Self, String> {
        let summary = summary.into();
        if summary.trim().is_empty() {
            return Err("plan summary must not be empty".to_owned());
        }
        if actions.is_empty() {
            return Err("plan must contain at least one action".to_owned());
        }
        for action in &actions {
            action.tool.validate()?;
        }

        Ok(Self {
            id: Uuid::new_v4(),
            summary,
            actions,
            created_at,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_local_speech() {
        let result = ActionProposal::new(
            "Speak",
            ToolCall::LocalSpeech {
                text: " ".to_owned(),
            },
            vec![],
            Utc::now(),
        );

        assert!(result.is_err());
    }
}
