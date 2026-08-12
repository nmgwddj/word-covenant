//! Native macOS microphone input backed by CPAL/CoreAudio.
//!
//! The CPAL callback has a deliberately narrow contract: normalize samples and
//! attempt to place them in [`CaptureIngress`]. It does not lock application
//! state, emit Tauri events, persist data, or run inference.

use super::{CaptureIngress, CapturePoint, MacOsInputDevice};
use block2::RcBlock;
use chrono::Utc;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample, Stream, StreamConfig};
use mach2::mach_time::{mach_continuous_time, mach_timebase_info, mach_timebase_info_data_t};
use objc2::runtime::Bool;
use objc2_av_foundation::{AVAuthorizationStatus, AVCaptureDevice, AVMediaType, AVMediaTypeAudio};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{mpsc, Arc, OnceLock};
use std::time::Duration;

const PERMISSION_WAIT_TIMEOUT: Duration = Duration::from_secs(90);
const MINIMUM_DBFS: f32 = -96.0;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MicrophonePermission {
    NotDetermined,
    Granted,
    Denied,
    Restricted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpalInputFailure {
    None,
    DeviceUnavailable,
    PermissionDenied,
    StreamInvalidated,
    Other,
}

#[derive(Clone, Copy, Debug)]
pub struct CpalInputTelemetry {
    pub rms_dbfs: f32,
    pub peak_dbfs: f32,
    pub clipping: bool,
    pub dropped_packets: u64,
    pub failure: CpalInputFailure,
}

#[derive(Clone, Debug)]
pub struct InputDeviceList {
    pub devices: Vec<MacOsInputDevice>,
    pub default_uid: Option<String>,
}

pub struct CpalInput {
    device: MacOsInputDevice,
    sample_rate: u32,
    channels: u16,
    stream: Option<Stream>,
    ingress: Arc<CaptureIngress>,
    telemetry: Arc<CaptureTelemetryAtomic>,
}

struct CaptureTelemetryAtomic {
    failure: AtomicU8,
}

impl CaptureTelemetryAtomic {
    fn new() -> Self {
        Self {
            failure: AtomicU8::new(CpalInputFailure::None as u8),
        }
    }

    fn snapshot(&self, dropped_packets: u64) -> CpalInputTelemetry {
        CpalInputTelemetry {
            // Level projection belongs to the native dispatcher, the sole
            // consumer of CaptureIngress. Keep this compatibility shape until
            // the service reads that projection directly.
            rms_dbfs: MINIMUM_DBFS,
            peak_dbfs: MINIMUM_DBFS,
            clipping: false,
            dropped_packets,
            failure: match self.failure.load(Ordering::Relaxed) {
                value if value == CpalInputFailure::DeviceUnavailable as u8 => {
                    CpalInputFailure::DeviceUnavailable
                }
                value if value == CpalInputFailure::PermissionDenied as u8 => {
                    CpalInputFailure::PermissionDenied
                }
                value if value == CpalInputFailure::StreamInvalidated as u8 => {
                    CpalInputFailure::StreamInvalidated
                }
                value if value == CpalInputFailure::Other as u8 => CpalInputFailure::Other,
                _ => CpalInputFailure::None,
            },
        }
    }

    fn record_failure(&self, failure: CpalInputFailure) {
        self.failure.store(failure as u8, Ordering::Relaxed);
    }
}

impl CpalInput {
    pub fn list_input_devices() -> Result<InputDeviceList, String> {
        let host = cpal::default_host();
        let default_device = host
            .default_input_device()
            .and_then(|device| input_device_metadata(&device).ok());
        match host.input_devices() {
            Ok(devices) => Ok(input_device_list_from_candidates(
                default_device,
                devices.map(|device| input_device_metadata(&device)),
            )),
            // CoreAudio's full directory can briefly fail during route changes
            // while a readable default device is still available. Keep that
            // device visible instead of presenting an empty selector.
            Err(_error) if default_device.is_some() => {
                Ok(input_device_list_from_candidates(default_device, []))
            }
            Err(error) => Err(format!("could not enumerate input devices: {error}")),
        }
    }

    pub fn permission_status() -> Result<MicrophonePermission, String> {
        let media_type = audio_media_type()?;
        let status = unsafe { AVCaptureDevice::authorizationStatusForMediaType(media_type) };
        Ok(permission_from_status(status))
    }

    /// Resolve macOS microphone permission before opening any input device.
    pub fn request_permission() -> Result<MicrophonePermission, String> {
        match Self::permission_status()? {
            MicrophonePermission::NotDetermined => {
                let media_type = audio_media_type()?;
                let (sender, receiver) = mpsc::sync_channel(1);
                let completion = RcBlock::new(move |granted: Bool| {
                    let _ = sender.send(granted.as_bool());
                });
                unsafe {
                    AVCaptureDevice::requestAccessForMediaType_completionHandler(
                        media_type,
                        &completion,
                    );
                }
                match receiver.recv_timeout(PERMISSION_WAIT_TIMEOUT) {
                    Ok(true) => Ok(MicrophonePermission::Granted),
                    Ok(false) => {
                        Ok(Self::permission_status().unwrap_or(MicrophonePermission::Denied))
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => Err(
                        "microphone permission request did not complete before the timeout"
                            .to_owned(),
                    ),
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        Err("microphone permission request was interrupted".to_owned())
                    }
                }
            }
            status => Ok(status),
        }
    }

    /// Resolve microphone access and prepare a paused input stream.
    ///
    /// The caller must attach the sole ingress dispatcher before calling
    /// [`Self::activate`], so no other native worker can consume PCM first.
    pub fn prepare(selected_device_uid: Option<&str>) -> Result<(Self, CapturePoint), String> {
        let permission = Self::request_permission()?;
        if permission != MicrophonePermission::Granted {
            return Err(match permission {
                MicrophonePermission::Denied => "microphone permission is denied".to_owned(),
                MicrophonePermission::Restricted => {
                    "microphone permission is restricted by macOS".to_owned()
                }
                MicrophonePermission::NotDetermined => {
                    "microphone permission is still undetermined".to_owned()
                }
                MicrophonePermission::Granted => unreachable!(),
            });
        }

        let host = cpal::default_host();
        // Resolve the hardware handle once. Re-enumerating separately for
        // metadata, configuration, and stream creation races unplug/replug
        // events and can bind configuration to a different route.
        let stream_device = select_input_device(&host, selected_device_uid)?;
        let device = input_device_metadata(&stream_device)?;
        let supported_config = device_config(&stream_device)?;
        let sample_format = supported_config.sample_format();
        let config = supported_config.config();
        let sample_rate = config.sample_rate;
        let channels = config.channels;
        let ingress = CaptureIngress::default_sized();
        let telemetry = Arc::new(CaptureTelemetryAtomic::new());
        let next_sample_offset = Arc::new(AtomicU64::new(0));
        let stream = build_stream(
            stream_device,
            config,
            sample_format,
            Arc::clone(&ingress),
            Arc::clone(&telemetry),
            next_sample_offset,
        )?;
        let anchor = capture_point_now();

        Ok((
            Self {
                device,
                sample_rate,
                channels,
                stream: Some(stream),
                ingress,
                telemetry,
            },
            anchor,
        ))
    }

    /// Start the prepared CoreAudio stream after its downstream dispatcher is
    /// ready to consume ingress packets.
    pub fn activate(&mut self) -> Result<(), String> {
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| "microphone input stream has already stopped".to_owned())?;
        stream
            .play()
            .map_err(|error| format!("could not start input stream: {error}"))
    }

    /// Backward-compatible single-phase start for callers that do not need to
    /// install a dispatcher between stream construction and activation.
    pub fn start(selected_device_uid: Option<&str>) -> Result<(Self, CapturePoint), String> {
        let (mut input, anchor) = Self::prepare(selected_device_uid)?;
        input.activate()?;
        Ok((input, anchor))
    }

    pub fn device(&self) -> &MacOsInputDevice {
        &self.device
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> u16 {
        self.channels
    }

    pub fn telemetry(&self) -> CpalInputTelemetry {
        self.telemetry.snapshot(self.ingress.dropped_packets())
    }

    pub(crate) fn ingress(&self) -> Arc<CaptureIngress> {
        Arc::clone(&self.ingress)
    }

    pub fn stop(&mut self) {
        release_stream(&mut self.stream);
    }
}

impl Drop for CpalInput {
    fn drop(&mut self) {
        self.stop();
    }
}

fn release_stream<T>(stream: &mut Option<T>) {
    drop(stream.take());
}

fn build_stream(
    device: cpal::Device,
    config: StreamConfig,
    sample_format: SampleFormat,
    ingress: Arc<CaptureIngress>,
    telemetry: Arc<CaptureTelemetryAtomic>,
    next_sample_offset: Arc<AtomicU64>,
) -> Result<Stream, String> {
    match sample_format {
        SampleFormat::I8 => {
            build_typed_stream::<i8>(device, config, ingress, telemetry, next_sample_offset)
        }
        SampleFormat::I16 => {
            build_typed_stream::<i16>(device, config, ingress, telemetry, next_sample_offset)
        }
        SampleFormat::I24 => {
            build_typed_stream::<cpal::I24>(device, config, ingress, telemetry, next_sample_offset)
        }
        SampleFormat::I32 => {
            build_typed_stream::<i32>(device, config, ingress, telemetry, next_sample_offset)
        }
        SampleFormat::I64 => {
            build_typed_stream::<i64>(device, config, ingress, telemetry, next_sample_offset)
        }
        SampleFormat::U8 => {
            build_typed_stream::<u8>(device, config, ingress, telemetry, next_sample_offset)
        }
        SampleFormat::U16 => {
            build_typed_stream::<u16>(device, config, ingress, telemetry, next_sample_offset)
        }
        SampleFormat::U24 => {
            build_typed_stream::<cpal::U24>(device, config, ingress, telemetry, next_sample_offset)
        }
        SampleFormat::U32 => {
            build_typed_stream::<u32>(device, config, ingress, telemetry, next_sample_offset)
        }
        SampleFormat::U64 => {
            build_typed_stream::<u64>(device, config, ingress, telemetry, next_sample_offset)
        }
        SampleFormat::F32 => {
            build_typed_stream::<f32>(device, config, ingress, telemetry, next_sample_offset)
        }
        SampleFormat::F64 => {
            build_typed_stream::<f64>(device, config, ingress, telemetry, next_sample_offset)
        }
        unsupported => Err(format!("unsupported input sample format: {unsupported}")),
    }
}

fn build_typed_stream<T>(
    device: cpal::Device,
    config: StreamConfig,
    ingress: Arc<CaptureIngress>,
    telemetry: Arc<CaptureTelemetryAtomic>,
    next_sample_offset: Arc<AtomicU64>,
) -> Result<Stream, String>
where
    T: SizedSample + Copy + Send + 'static,
    f32: FromSample<T>,
{
    let channels = config.channels;
    let sample_rate = config.sample_rate;
    let callback_ingress = Arc::clone(&ingress);
    let callback_offset = Arc::clone(&next_sample_offset);
    let error_telemetry = telemetry;

    device
        .build_input_stream(
            config,
            move |samples: &[T], _| {
                let frames = samples.len() / usize::from(channels);
                let starting_sample_offset = callback_offset
                    .fetch_add(u64::try_from(frames).unwrap_or(u64::MAX), Ordering::Relaxed);
                let _ = callback_ingress.try_write_mapped(
                    starting_sample_offset,
                    sample_rate,
                    channels,
                    samples,
                    <f32 as Sample>::from_sample,
                );
            },
            move |error| error_telemetry.record_failure(cpal_failure(error.kind())),
            None,
        )
        .map_err(|error| format!("could not build input stream: {error}"))
}

fn select_input_device(
    host: &cpal::Host,
    selected_device_uid: Option<&str>,
) -> Result<cpal::Device, String> {
    let Some(selected_device_uid) = selected_device_uid else {
        return host
            .default_input_device()
            .ok_or_else(|| "no default microphone is available".to_owned());
    };

    for device in host
        .input_devices()
        .map_err(|error| format!("could not enumerate input devices: {error}"))?
    {
        if let Ok(uid) = device_uid(&device) {
            if uid == selected_device_uid {
                return Ok(device);
            }
        }
    }

    Err(format!(
        "selected input device {selected_device_uid:?} is unavailable"
    ))
}

fn device_config(device: &cpal::Device) -> Result<cpal::SupportedStreamConfig, String> {
    device
        .default_input_config()
        .map_err(|error| format!("could not get default input stream configuration: {error}"))
}

fn input_device_metadata(device: &cpal::Device) -> Result<MacOsInputDevice, String> {
    let uid = device_uid(device)?;
    let name = device
        .description()
        .map(|description| description.name().to_owned())
        .map_err(|error| format!("could not read input device name: {error}"))?;
    MacOsInputDevice::new(uid, name).map_err(|error| error.to_string())
}

fn input_device_list_from_candidates(
    default_device: Option<MacOsInputDevice>,
    candidates: impl IntoIterator<Item = Result<MacOsInputDevice, String>>,
) -> InputDeviceList {
    let default_uid = default_device
        .as_ref()
        .map(|device| device.uid().to_owned());
    let mut devices: Vec<MacOsInputDevice> = Vec::new();

    for device in candidates.into_iter().flatten() {
        if !devices.iter().any(|known| known.uid() == device.uid()) {
            devices.push(device);
        }
    }
    if let Some(default_device) = default_device {
        if !devices
            .iter()
            .any(|known| known.uid() == default_device.uid())
        {
            devices.push(default_device);
        }
    }

    InputDeviceList {
        devices,
        default_uid,
    }
}

fn device_uid(device: &cpal::Device) -> Result<String, String> {
    device
        .id()
        .map(|id| id.to_string())
        .map_err(|error| format!("could not read input device UID: {error}"))
}

pub fn capture_point_now() -> CapturePoint {
    CapturePoint {
        monotonic_ns: continuous_time_ns(),
        wall_clock: Utc::now(),
    }
}

fn audio_media_type() -> Result<&'static AVMediaType, String> {
    unsafe { AVMediaTypeAudio }
        .ok_or_else(|| "AVFoundation does not expose the audio media type".to_owned())
}

fn permission_from_status(status: AVAuthorizationStatus) -> MicrophonePermission {
    match status {
        AVAuthorizationStatus::Authorized => MicrophonePermission::Granted,
        AVAuthorizationStatus::Denied => MicrophonePermission::Denied,
        AVAuthorizationStatus::Restricted => MicrophonePermission::Restricted,
        _ => MicrophonePermission::NotDetermined,
    }
}

fn cpal_failure(error: cpal::ErrorKind) -> CpalInputFailure {
    match error {
        cpal::ErrorKind::DeviceNotAvailable => CpalInputFailure::DeviceUnavailable,
        cpal::ErrorKind::PermissionDenied => CpalInputFailure::PermissionDenied,
        cpal::ErrorKind::StreamInvalidated => CpalInputFailure::StreamInvalidated,
        _ => CpalInputFailure::Other,
    }
}

fn continuous_time_ns() -> u64 {
    let timebase = *TIMEBASE.get_or_init(|| {
        let mut timebase = mach_timebase_info_data_t { numer: 0, denom: 0 };
        let result = unsafe { mach_timebase_info(&mut timebase) };
        assert_eq!(result, 0, "mach timebase information is available on macOS");
        timebase
    });
    let ticks = unsafe { mach_continuous_time() };
    (u128::from(ticks) * u128::from(timebase.numer) / u128::from(timebase.denom))
        .min(u128::from(u64::MAX)) as u64
}

static TIMEBASE: OnceLock<mach_timebase_info_data_t> = OnceLock::new();

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    #[test]
    fn releasing_the_stream_handle_drops_its_resource() {
        struct DropProbe(Rc<Cell<bool>>);

        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.set(true);
            }
        }

        let dropped = Rc::new(Cell::new(false));
        let mut stream = Some(DropProbe(Rc::clone(&dropped)));

        release_stream(&mut stream);

        assert!(stream.is_none());
        assert!(dropped.get());
    }

    #[test]
    fn continuous_clock_is_non_decreasing() {
        let first = continuous_time_ns();
        let second = continuous_time_ns();
        assert!(second >= first);
    }

    #[test]
    fn device_directory_skips_individually_unreadable_devices() {
        let default = MacOsInputDevice::new("built-in", "MacBook microphone").unwrap();
        let usb = MacOsInputDevice::new("usb", "USB microphone").unwrap();

        let directory = input_device_list_from_candidates(
            Some(default.clone()),
            vec![
                Ok(default.clone()),
                Err("could not read transient device identity".to_owned()),
                Ok(usb.clone()),
            ],
        );

        assert_eq!(directory.default_uid.as_deref(), Some(default.uid()));
        assert_eq!(directory.devices, vec![default, usb]);
    }

    #[test]
    fn device_directory_retains_a_readable_default_when_enumeration_is_empty() {
        let default = MacOsInputDevice::new("built-in", "MacBook microphone").unwrap();

        let directory = input_device_list_from_candidates(Some(default.clone()), Vec::new());

        assert_eq!(directory.default_uid.as_deref(), Some(default.uid()));
        assert_eq!(directory.devices, vec![default]);
    }
}
