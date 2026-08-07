use crate::domain::DataCategory;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use url::Url;
use uuid::Uuid;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EgressApproval {
    pub id: Uuid,
    pub tool_id: String,
    pub origin: String,
    pub data_categories: BTreeSet<DataCategory>,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub revoked_at: Option<DateTime<Utc>>,
}

impl EgressApproval {
    pub fn new(
        tool_id: impl Into<String>,
        origin: impl AsRef<str>,
        data_categories: BTreeSet<DataCategory>,
        created_at: DateTime<Utc>,
        expires_at: Option<DateTime<Utc>>,
    ) -> Result<Self, PolicyReason> {
        let tool_id = tool_id.into();
        if tool_id.trim().is_empty() {
            return Err(PolicyReason::MissingToolIdentifier);
        }
        let origin = normalize_https_origin(origin.as_ref())?;

        Ok(Self {
            id: Uuid::new_v4(),
            tool_id,
            origin,
            data_categories,
            created_at,
            expires_at,
            revoked_at: None,
        })
    }

    pub fn revoke(&mut self, revoked_at: DateTime<Utc>) {
        self.revoked_at = Some(revoked_at);
    }

    fn is_active_at(&self, now: DateTime<Utc>) -> bool {
        self.revoked_at.is_none() && self.expires_at.is_none_or(|expires_at| expires_at > now)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EgressRequest {
    pub tool_id: String,
    pub origin: String,
    pub data_categories: BTreeSet<DataCategory>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyReason {
    EgressDisabled,
    DeniedByDefault,
    MissingToolIdentifier,
    InvalidOrigin,
    InsecureOrigin,
    OriginMismatch,
    ApprovalExpired,
    ApprovalRevoked,
    DataScopeMismatch,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum PolicyDecision {
    Allowed { approval_id: Uuid },
    Denied { reason: PolicyReason },
}

#[derive(Clone, Debug, Default)]
pub struct EgressPolicy {
    egress_enabled: bool,
    approvals: Vec<EgressApproval>,
}

impl EgressPolicy {
    pub fn new(approvals: Vec<EgressApproval>) -> Self {
        Self {
            egress_enabled: false,
            approvals,
        }
    }

    pub fn egress_enabled(&self) -> bool {
        self.egress_enabled
    }

    pub fn set_egress_enabled(&mut self, enabled: bool) -> bool {
        let changed = self.egress_enabled != enabled;
        self.egress_enabled = enabled;
        changed
    }

    pub fn approvals(&self) -> &[EgressApproval] {
        &self.approvals
    }

    pub fn add_approval(&mut self, approval: EgressApproval) {
        self.approvals.push(approval);
    }

    pub fn revoke(&mut self, approval_id: Uuid, revoked_at: DateTime<Utc>) -> bool {
        let Some(approval) = self
            .approvals
            .iter_mut()
            .find(|approval| approval.id == approval_id)
        else {
            return false;
        };
        approval.revoke(revoked_at);
        true
    }

    pub fn evaluate(&self, request: &EgressRequest, now: DateTime<Utc>) -> PolicyDecision {
        if !self.egress_enabled {
            return PolicyDecision::Denied {
                reason: PolicyReason::EgressDisabled,
            };
        }

        let origin = match normalize_https_origin(&request.origin) {
            Ok(origin) => origin,
            Err(reason) => return PolicyDecision::Denied { reason },
        };

        let matching_tool: Vec<&EgressApproval> = self
            .approvals
            .iter()
            .filter(|approval| approval.tool_id == request.tool_id)
            .collect();

        if matching_tool.is_empty() {
            return PolicyDecision::Denied {
                reason: PolicyReason::DeniedByDefault,
            };
        }

        let matching_origin: Vec<&EgressApproval> = matching_tool
            .into_iter()
            .filter(|approval| approval.origin == origin)
            .collect();

        if matching_origin.is_empty() {
            return PolicyDecision::Denied {
                reason: PolicyReason::OriginMismatch,
            };
        }

        if matching_origin
            .iter()
            .all(|approval| approval.revoked_at.is_some())
        {
            return PolicyDecision::Denied {
                reason: PolicyReason::ApprovalRevoked,
            };
        }

        if matching_origin
            .iter()
            .filter(|approval| approval.revoked_at.is_none())
            .all(|approval| {
                approval
                    .expires_at
                    .is_some_and(|expires_at| expires_at <= now)
            })
        {
            return PolicyDecision::Denied {
                reason: PolicyReason::ApprovalExpired,
            };
        }

        if let Some(approval) = matching_origin.into_iter().find(|approval| {
            approval.is_active_at(now)
                && approval
                    .data_categories
                    .is_superset(&request.data_categories)
        }) {
            return PolicyDecision::Allowed {
                approval_id: approval.id,
            };
        }

        PolicyDecision::Denied {
            reason: PolicyReason::DataScopeMismatch,
        }
    }
}

fn normalize_https_origin(value: &str) -> Result<String, PolicyReason> {
    let url = Url::parse(value).map_err(|_| PolicyReason::InvalidOrigin)?;
    if url.scheme() != "https" {
        return Err(PolicyReason::InsecureOrigin);
    }
    if url.host_str().is_none() || !url.username().is_empty() || url.password().is_some() {
        return Err(PolicyReason::InvalidOrigin);
    }

    Ok(url.origin().ascii_serialization())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn categories(items: &[DataCategory]) -> BTreeSet<DataCategory> {
        items.iter().cloned().collect()
    }

    fn request(origin: &str) -> EgressRequest {
        EgressRequest {
            tool_id: "crm-sync".to_owned(),
            origin: origin.to_owned(),
            data_categories: categories(&[DataCategory::Summary]),
        }
    }

    #[test]
    fn denies_network_when_the_master_switch_is_off() {
        let decision =
            EgressPolicy::default().evaluate(&request("https://api.example.com"), Utc::now());

        assert_eq!(
            decision,
            PolicyDecision::Denied {
                reason: PolicyReason::EgressDisabled,
            }
        );
    }

    #[test]
    fn still_denies_by_default_after_the_master_switch_is_enabled() {
        let mut policy = EgressPolicy::default();
        assert!(policy.set_egress_enabled(true));

        assert_eq!(
            policy.evaluate(&request("https://api.example.com"), Utc::now()),
            PolicyDecision::Denied {
                reason: PolicyReason::DeniedByDefault,
            }
        );
    }

    #[test]
    fn requires_the_exact_approved_origin_and_data_scope() {
        let now = Utc::now();
        let approval = EgressApproval::new(
            "crm-sync",
            "https://api.example.com/v1/records",
            categories(&[DataCategory::Summary]),
            now,
            None,
        )
        .unwrap();
        let mut policy = EgressPolicy::new(vec![approval]);
        policy.set_egress_enabled(true);

        assert!(matches!(
            policy.evaluate(&request("https://api.example.com/another-path"), now),
            PolicyDecision::Allowed { .. }
        ));
        assert_eq!(
            policy.evaluate(&request("https://other.example.com"), now),
            PolicyDecision::Denied {
                reason: PolicyReason::OriginMismatch,
            }
        );
    }

    #[test]
    fn rejects_expired_or_insecure_approval_paths() {
        let now = Utc::now();
        let approval = EgressApproval::new(
            "crm-sync",
            "https://api.example.com",
            categories(&[DataCategory::Summary]),
            now,
            Some(now),
        )
        .unwrap();
        let mut policy = EgressPolicy::new(vec![approval]);
        policy.set_egress_enabled(true);

        assert_eq!(
            policy.evaluate(&request("https://api.example.com"), now),
            PolicyDecision::Denied {
                reason: PolicyReason::ApprovalExpired,
            }
        );
        assert_eq!(
            policy.evaluate(&request("http://api.example.com"), now),
            PolicyDecision::Denied {
                reason: PolicyReason::InsecureOrigin,
            }
        );
    }
}
