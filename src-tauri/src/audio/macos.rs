//! macOS input-capture boundary.
//!
//! This module deliberately contains no CoreAudio or `cpal` dependency yet. A
//! future backend should translate its input callback and device-listener
//! notifications into the values in this module, then forward them to
//! [`MacOsCaptureAdapter`]. Keeping that translation at this boundary lets the
//! clock, queue, and lifecycle behavior be tested without a microphone.

use super::{BoundedCaptureWriter, CaptureClock, CapturePacket, CapturePoint, CaptureWriteResult};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt;

/// The stable UID exposed by CoreAudio for an input device plus its display
/// name. The UID, rather than the name, is used to match callbacks to a
/// running stream because names can change or be duplicated.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MacOsInputDevice {
    uid: String,
    name: String,
}

impl MacOsInputDevice {
    pub fn new(uid: impl Into<String>, name: impl Into<String>) -> Result<Self, MacOsCaptureError> {
        let uid = uid.into();
        let name = name.into();
        if uid.trim().is_empty() {
            return Err(MacOsCaptureError::InvalidDeviceIdentity("uid"));
        }
        if name.trim().is_empty() {
            return Err(MacOsCaptureError::InvalidDeviceIdentity("name"));
        }
        Ok(Self { uid, name })
    }

    pub fn uid(&self) -> &str {
        &self.uid
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

/// A normalized input callback payload.
///
/// The CoreAudio implementation should convert `AudioTimeStamp::mHostTime`
/// (or the equivalent `cpal` callback timing) into `session_anchor` and a
/// sample offset before constructing this value. `CaptureClock` then derives
/// wall-clock timestamps from that stable anchor. The adapter owns no native
/// handles and can therefore be exercised entirely with deterministic data.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MacOsInputCallback {
    pub device_uid: String,
    pub session_anchor: CapturePoint,
    pub starting_sample_offset: u64,
    pub sample_rate: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
}

impl MacOsInputCallback {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device_uid: impl Into<String>,
        session_anchor: CapturePoint,
        starting_sample_offset: u64,
        sample_rate: u32,
        channels: u16,
        samples: Vec<f32>,
    ) -> Result<Self, MacOsCaptureError> {
        let callback = Self {
            device_uid: device_uid.into(),
            session_anchor,
            starting_sample_offset,
            sample_rate,
            channels,
            samples,
        };
        callback.validate()?;
        Ok(callback)
    }

    /// Return the timestamp of the first sample in this callback.
    pub fn packet_start(&self) -> Result<CapturePoint, MacOsCaptureError> {
        self.validate()?;
        let clock = CaptureClock::new(self.session_anchor.clone(), self.sample_rate)
            .map_err(MacOsCaptureError::InvalidCallback)?;
        Ok(clock.point_at_sample_offset(self.starting_sample_offset))
    }

    /// Convert the normalized callback into the queue packet used by the
    /// existing capture pipeline.
    pub fn into_packet(self) -> Result<CapturePacket, MacOsCaptureError> {
        self.validate()?;
        Ok(CapturePacket {
            starting_sample_offset: self.starting_sample_offset,
            sample_rate: self.sample_rate,
            channels: self.channels,
            samples: self.samples,
        })
    }

    fn validate(&self) -> Result<(), MacOsCaptureError> {
        if self.device_uid.trim().is_empty() {
            return Err(MacOsCaptureError::InvalidCallback(
                "device uid must not be empty".to_owned(),
            ));
        }
        if self.sample_rate == 0 {
            return Err(MacOsCaptureError::InvalidCallback(
                "sample rate must be greater than zero".to_owned(),
            ));
        }
        if self.channels == 0 {
            return Err(MacOsCaptureError::InvalidCallback(
                "channel count must be greater than zero".to_owned(),
            ));
        }
        if !self
            .samples
            .len()
            .is_multiple_of(usize::from(self.channels))
        {
            return Err(MacOsCaptureError::InvalidCallback(
                "sample buffer must contain complete interleaved frames".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Lifecycle and reliability events emitted by the adapter.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(
    tag = "kind",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum MacOsCaptureEvent {
    CaptureStarted {
        device: MacOsInputDevice,
        at: CapturePoint,
    },
    InputDeviceChanged {
        previous_device: MacOsInputDevice,
        current_device: MacOsInputDevice,
        at: CapturePoint,
    },
    InputDeviceUnavailable {
        device: MacOsInputDevice,
        at: CapturePoint,
    },
    CaptureStopped {
        device: MacOsInputDevice,
        at: CapturePoint,
    },
    PacketDropped {
        device: MacOsInputDevice,
        at: CapturePoint,
        starting_sample_offset: u64,
    },
    CaptureQueueClosed {
        device: MacOsInputDevice,
        at: CapturePoint,
        starting_sample_offset: u64,
    },
}

/// Errors raised at the callback/lifecycle boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MacOsCaptureError {
    AlreadyRunning,
    NotRunning,
    InvalidDeviceIdentity(&'static str),
    InvalidCallback(String),
    DeviceUidMismatch { expected: String, received: String },
}

impl fmt::Display for MacOsCaptureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyRunning => formatter.write_str("capture is already running"),
            Self::NotRunning => formatter.write_str("capture is not running"),
            Self::InvalidDeviceIdentity(field) => {
                write!(formatter, "input device {field} must not be empty")
            }
            Self::InvalidCallback(message) => formatter.write_str(message),
            Self::DeviceUidMismatch { expected, received } => write!(
                formatter,
                "callback device uid {received:?} does not match active device {expected:?}"
            ),
        }
    }
}

impl std::error::Error for MacOsCaptureError {}

/// Boundary implemented by the future `cpal`/CoreAudio backend.
///
/// An input stream callback should call [`Self::on_input_callback`] after
/// translating its timing metadata. A CoreAudio device-property listener
/// should call the device methods when the default input changes or becomes
/// unavailable. No native framework types cross this trait boundary.
pub trait MacOsCaptureCallbackSink {
    fn on_input_callback(
        &mut self,
        callback: MacOsInputCallback,
    ) -> Result<CaptureWriteResult, MacOsCaptureError>;

    fn on_input_device_changed(
        &mut self,
        device: MacOsInputDevice,
        observed_at: CapturePoint,
    ) -> Result<(), MacOsCaptureError>;

    fn on_input_device_unavailable(
        &mut self,
        device_uid: &str,
        observed_at: CapturePoint,
    ) -> Result<(), MacOsCaptureError>;
}

/// Callback-driven producer that feeds the existing bounded capture queue.
///
/// This is intentionally a small state machine rather than a microphone
/// implementation. It gives the native backend a single place to enforce
/// device identity, preserve sample-clock timestamps, and expose loss events.
pub struct MacOsCaptureAdapter {
    writer: BoundedCaptureWriter,
    active_device: Option<MacOsInputDevice>,
    running: bool,
    events: VecDeque<MacOsCaptureEvent>,
}

impl MacOsCaptureAdapter {
    pub fn new(writer: BoundedCaptureWriter) -> Self {
        Self {
            writer,
            active_device: None,
            running: false,
            events: VecDeque::new(),
        }
    }

    /// Mark a native input stream as active. The timestamp should come from
    /// the same host-clock conversion used by subsequent callbacks.
    pub fn start(
        &mut self,
        device: MacOsInputDevice,
        observed_at: CapturePoint,
    ) -> Result<(), MacOsCaptureError> {
        if self.running {
            return Err(MacOsCaptureError::AlreadyRunning);
        }
        self.active_device = Some(device.clone());
        self.running = true;
        self.events.push_back(MacOsCaptureEvent::CaptureStarted {
            device,
            at: observed_at,
        });
        Ok(())
    }

    pub fn stop(&mut self, observed_at: CapturePoint) -> Result<(), MacOsCaptureError> {
        if !self.running {
            return Err(MacOsCaptureError::NotRunning);
        }
        let device = self
            .active_device
            .take()
            .ok_or(MacOsCaptureError::NotRunning)?;
        self.running = false;
        self.events.push_back(MacOsCaptureEvent::CaptureStopped {
            device,
            at: observed_at,
        });
        Ok(())
    }

    pub fn is_running(&self) -> bool {
        self.running
    }

    pub fn active_device(&self) -> Option<&MacOsInputDevice> {
        self.active_device.as_ref()
    }

    pub fn try_next_event(&mut self) -> Option<MacOsCaptureEvent> {
        self.events.pop_front()
    }
}

impl MacOsCaptureCallbackSink for MacOsCaptureAdapter {
    fn on_input_callback(
        &mut self,
        callback: MacOsInputCallback,
    ) -> Result<CaptureWriteResult, MacOsCaptureError> {
        let device = self
            .active_device
            .clone()
            .ok_or(MacOsCaptureError::NotRunning)?;
        if !self.running {
            return Err(MacOsCaptureError::NotRunning);
        }
        if callback.device_uid != device.uid {
            return Err(MacOsCaptureError::DeviceUidMismatch {
                expected: device.uid,
                received: callback.device_uid,
            });
        }

        let packet_start = callback.packet_start()?;
        let starting_sample_offset = callback.starting_sample_offset;
        let write_result = self.writer.try_write(callback.into_packet()?);
        match write_result {
            CaptureWriteResult::Dropped => {
                self.events.push_back(MacOsCaptureEvent::PacketDropped {
                    device,
                    at: packet_start,
                    starting_sample_offset,
                })
            }
            CaptureWriteResult::Closed => {
                self.events
                    .push_back(MacOsCaptureEvent::CaptureQueueClosed {
                        device,
                        at: packet_start,
                        starting_sample_offset,
                    })
            }
            CaptureWriteResult::Enqueued => {}
        }
        Ok(write_result)
    }

    fn on_input_device_changed(
        &mut self,
        device: MacOsInputDevice,
        observed_at: CapturePoint,
    ) -> Result<(), MacOsCaptureError> {
        if !self.running {
            return Err(MacOsCaptureError::NotRunning);
        }
        let previous_device = self
            .active_device
            .as_ref()
            .ok_or(MacOsCaptureError::NotRunning)?;
        if previous_device.uid == device.uid {
            // A CoreAudio listener can report a metadata-only update. Keep
            // the latest display name without fabricating a device change.
            self.active_device = Some(device);
            return Ok(());
        }
        let previous_device = previous_device.clone();
        self.active_device = Some(device.clone());
        self.events
            .push_back(MacOsCaptureEvent::InputDeviceChanged {
                previous_device,
                current_device: device,
                at: observed_at,
            });
        Ok(())
    }

    fn on_input_device_unavailable(
        &mut self,
        device_uid: &str,
        observed_at: CapturePoint,
    ) -> Result<(), MacOsCaptureError> {
        if !self.running {
            return Err(MacOsCaptureError::NotRunning);
        }
        let device = self
            .active_device
            .take()
            .ok_or(MacOsCaptureError::NotRunning)?;
        if device.uid != device_uid {
            let expected = device.uid.clone();
            self.active_device = Some(device);
            return Err(MacOsCaptureError::DeviceUidMismatch {
                expected,
                received: device_uid.to_owned(),
            });
        }
        self.running = false;
        self.events
            .push_back(MacOsCaptureEvent::InputDeviceUnavailable {
                device,
                at: observed_at,
            });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, Duration, Utc};
    use std::sync::mpsc::sync_channel;

    fn point(monotonic_ns: u64, milliseconds: i64) -> CapturePoint {
        CapturePoint {
            monotonic_ns,
            wall_clock: DateTime::<Utc>::UNIX_EPOCH + Duration::milliseconds(milliseconds),
        }
    }

    fn device(uid: &str) -> MacOsInputDevice {
        MacOsInputDevice::new(uid, format!("{uid} microphone")).unwrap()
    }

    fn callback(device_uid: &str, offset: u64, samples: Vec<f32>) -> MacOsInputCallback {
        MacOsInputCallback::new(device_uid, point(100, 0), offset, 16_000, 1, samples).unwrap()
    }

    #[test]
    fn forwards_normalized_callback_into_the_existing_capture_queue() {
        let (sender, receiver) = sync_channel(1);
        let mut adapter = MacOsCaptureAdapter::new(BoundedCaptureWriter::new(sender));
        let input = device("built-in");
        adapter.start(input.clone(), point(100, 0)).unwrap();

        assert_eq!(
            adapter.try_next_event(),
            Some(MacOsCaptureEvent::CaptureStarted {
                device: input,
                at: point(100, 0),
            })
        );
        assert_eq!(
            adapter.on_input_callback(callback("built-in", 16_000, vec![0.1, 0.2])),
            Ok(CaptureWriteResult::Enqueued)
        );

        let packet = receiver.recv().unwrap();
        assert_eq!(packet.starting_sample_offset, 16_000);
        assert_eq!(packet.sample_rate, 16_000);
        assert_eq!(packet.samples, vec![0.1, 0.2]);
    }

    #[test]
    fn maps_the_callback_sample_offset_to_a_deterministic_timestamp() {
        let callback = callback("built-in", 8_000, vec![0.0; 8]);

        assert_eq!(callback.packet_start().unwrap(), point(500_000_100, 500));
    }

    #[test]
    fn emits_device_identity_change_and_rejects_stale_callbacks() {
        let (sender, receiver) = sync_channel(2);
        let mut adapter = MacOsCaptureAdapter::new(BoundedCaptureWriter::new(sender));
        let old_device = device("built-in");
        let new_device = device("usb-mic");
        adapter.start(old_device.clone(), point(0, 0)).unwrap();
        adapter.try_next_event();

        adapter
            .on_input_device_changed(new_device.clone(), point(10, 10))
            .unwrap();
        assert_eq!(
            adapter.try_next_event(),
            Some(MacOsCaptureEvent::InputDeviceChanged {
                previous_device: old_device,
                current_device: new_device.clone(),
                at: point(10, 10),
            })
        );

        let error = adapter
            .on_input_callback(callback("built-in", 0, vec![0.0; 2]))
            .unwrap_err();
        assert_eq!(
            error,
            MacOsCaptureError::DeviceUidMismatch {
                expected: "usb-mic".to_owned(),
                received: "built-in".to_owned(),
            }
        );
        assert_eq!(
            adapter.on_input_callback(callback("usb-mic", 0, vec![0.0; 2])),
            Ok(CaptureWriteResult::Enqueued)
        );
        assert_eq!(receiver.recv().unwrap().samples, vec![0.0; 2]);
    }

    #[test]
    fn records_a_queue_drop_at_the_callback_start_time() {
        let (sender, _receiver) = sync_channel(1);
        let mut adapter = MacOsCaptureAdapter::new(BoundedCaptureWriter::new(sender));
        adapter.start(device("built-in"), point(100, 0)).unwrap();
        adapter.try_next_event();

        assert_eq!(
            adapter.on_input_callback(callback("built-in", 0, vec![0.0])),
            Ok(CaptureWriteResult::Enqueued)
        );
        assert_eq!(
            adapter.on_input_callback(callback("built-in", 8_000, vec![0.0])),
            Ok(CaptureWriteResult::Dropped)
        );
        assert_eq!(
            adapter.try_next_event(),
            Some(MacOsCaptureEvent::PacketDropped {
                device: device("built-in"),
                at: point(500_000_100, 500),
                starting_sample_offset: 8_000,
            })
        );
    }

    #[test]
    fn unavailable_device_stops_callbacks_and_emits_an_event() {
        let (sender, _receiver) = sync_channel(1);
        let mut adapter = MacOsCaptureAdapter::new(BoundedCaptureWriter::new(sender));
        let input = device("built-in");
        adapter.start(input.clone(), point(0, 0)).unwrap();
        adapter.try_next_event();

        adapter
            .on_input_device_unavailable("built-in", point(20, 20))
            .unwrap();
        assert!(!adapter.is_running());
        assert_eq!(
            adapter.try_next_event(),
            Some(MacOsCaptureEvent::InputDeviceUnavailable {
                device: input,
                at: point(20, 20),
            })
        );
        assert_eq!(
            adapter.on_input_callback(callback("built-in", 0, vec![0.0])),
            Err(MacOsCaptureError::NotRunning)
        );
    }
}
