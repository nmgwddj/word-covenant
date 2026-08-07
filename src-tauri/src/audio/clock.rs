use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CapturePoint {
    pub monotonic_ns: u64,
    pub wall_clock: DateTime<Utc>,
}

#[derive(Clone, Debug)]
pub struct CaptureClock {
    anchor: CapturePoint,
    sample_rate: u32,
}

impl CaptureClock {
    pub fn new(anchor: CapturePoint, sample_rate: u32) -> Result<Self, String> {
        if sample_rate == 0 {
            return Err("sample rate must be greater than zero".to_owned());
        }
        Ok(Self {
            anchor,
            sample_rate,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn point_at_sample_offset(&self, sample_offset: u64) -> CapturePoint {
        let elapsed_ns = (u128::from(sample_offset) * 1_000_000_000_u128
            / u128::from(self.sample_rate))
        .min(u128::from(u64::MAX)) as u64;
        let elapsed_chrono_ns = elapsed_ns.min(i64::MAX as u64) as i64;

        CapturePoint {
            monotonic_ns: self.anchor.monotonic_ns.saturating_add(elapsed_ns),
            wall_clock: self.anchor.wall_clock + Duration::nanoseconds(elapsed_chrono_ns),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureGapReason {
    InputDeviceChanged,
    InputDeviceUnavailable,
    SystemSleep,
    QueueOverrun,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureGap {
    pub started_at: CapturePoint,
    pub ended_at: CapturePoint,
    pub reason: CaptureGapReason,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_time_from_the_sample_clock() {
        let anchor = CapturePoint {
            monotonic_ns: 1_000,
            wall_clock: DateTime::UNIX_EPOCH,
        };
        let clock = CaptureClock::new(anchor, 16_000).unwrap();

        let point = clock.point_at_sample_offset(16_000);

        assert_eq!(point.monotonic_ns, 1_000_000_000 + 1_000);
        assert_eq!(
            point.wall_clock,
            DateTime::UNIX_EPOCH + Duration::seconds(1)
        );
    }

    #[test]
    fn rejects_an_invalid_sample_rate() {
        let result = CaptureClock::new(
            CapturePoint {
                monotonic_ns: 0,
                wall_clock: Utc::now(),
            },
            0,
        );

        assert!(result.is_err());
    }
}
