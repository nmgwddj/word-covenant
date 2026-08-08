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
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::{mpsc, Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const PERMISSION_WAIT_TIMEOUT: Duration = Duration::from_secs(90);
const PCM_WORKER_IDLE_SLEEP: Duration = Duration::from_millis(2);
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
    stop_worker: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

struct CaptureTelemetryAtomic {
    rms_dbfs_bits: AtomicU32,
    peak_dbfs_bits: AtomicU32,
    clipping: AtomicBool,
    failure: AtomicU8,
}

impl CaptureTelemetryAtomic {
    fn new() -> Self {
        Self {
            rms_dbfs_bits: AtomicU32::new(MINIMUM_DBFS.to_bits()),
            peak_dbfs_bits: AtomicU32::new(MINIMUM_DBFS.to_bits()),
            clipping: AtomicBool::new(false),
            failure: AtomicU8::new(CpalInputFailure::None as u8),
        }
    }

    fn snapshot(&self, dropped_packets: u64) -> CpalInputTelemetry {
        CpalInputTelemetry {
            rms_dbfs: f32::from_bits(self.rms_dbfs_bits.load(Ordering::Relaxed)),
            peak_dbfs: f32::from_bits(self.peak_dbfs_bits.load(Ordering::Relaxed)),
            clipping: self.clipping.load(Ordering::Relaxed),
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

    fn record_level(&self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }

        let mut square_sum = 0.0_f64;
        let mut peak = 0.0_f32;
        let mut clipping = false;
        for sample in samples {
            let magnitude = sample.abs();
            square_sum += f64::from(*sample) * f64::from(*sample);
            peak = peak.max(magnitude);
            clipping |= magnitude >= 0.999;
        }
        let rms = (square_sum / samples.len() as f64).sqrt() as f32;
        self.rms_dbfs_bits
            .store(to_dbfs(rms).to_bits(), Ordering::Relaxed);
        self.peak_dbfs_bits
            .store(to_dbfs(peak).to_bits(), Ordering::Relaxed);
        self.clipping.store(clipping, Ordering::Relaxed);
    }

    fn record_failure(&self, failure: CpalInputFailure) {
        self.failure.store(failure as u8, Ordering::Relaxed);
    }
}

impl CpalInput {
    pub fn list_input_devices() -> Result<InputDeviceList, String> {
        let host = cpal::default_host();
        let default_uid = host
            .default_input_device()
            .map(|device| device_uid(&device))
            .transpose()?;
        let devices = host
            .input_devices()
            .map_err(|error| format!("could not enumerate input devices: {error}"))?
            .map(|device| {
                let uid = device_uid(&device)?;
                MacOsInputDevice::new(uid, device.to_string()).map_err(|error| error.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(InputDeviceList {
            devices,
            default_uid,
        })
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

    pub fn start(selected_device_uid: Option<&str>) -> Result<(Self, CapturePoint), String> {
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
        let device = select_input_device(&host, selected_device_uid)?;
        let device_uid = device_uid(&device)?;
        let device = MacOsInputDevice::new(device_uid, device.to_string())
            .map_err(|error| error.to_string())?;
        let supported_config = device_config(&host, selected_device_uid)?;
        let sample_format = supported_config.sample_format();
        let config = supported_config.config();
        let sample_rate = config.sample_rate;
        let channels = config.channels;
        let ingress = CaptureIngress::default_sized();
        let telemetry = Arc::new(CaptureTelemetryAtomic::new());
        let next_sample_offset = Arc::new(AtomicU64::new(0));
        let stream = build_stream(
            &host,
            selected_device_uid,
            config,
            sample_format,
            Arc::clone(&ingress),
            Arc::clone(&telemetry),
            next_sample_offset,
        )?;
        let stop_worker = Arc::new(AtomicBool::new(false));
        let worker = spawn_pcm_worker(
            Arc::clone(&ingress),
            Arc::clone(&telemetry),
            Arc::clone(&stop_worker),
        );
        let anchor = capture_point_now();

        if let Err(error) = stream.play() {
            stop_worker.store(true, Ordering::Release);
            let _ = worker.join();
            return Err(format!("could not start input stream: {error}"));
        }

        Ok((
            Self {
                device,
                sample_rate,
                channels,
                stream: Some(stream),
                ingress,
                telemetry,
                stop_worker,
                worker: Some(worker),
            },
            anchor,
        ))
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

    pub fn stop(&mut self) {
        self.stream.take();
        self.stop_worker.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Drop for CpalInput {
    fn drop(&mut self) {
        self.stop();
    }
}

fn build_stream(
    host: &cpal::Host,
    selected_device_uid: Option<&str>,
    config: StreamConfig,
    sample_format: SampleFormat,
    ingress: Arc<CaptureIngress>,
    telemetry: Arc<CaptureTelemetryAtomic>,
    next_sample_offset: Arc<AtomicU64>,
) -> Result<Stream, String> {
    let device = select_input_device(host, selected_device_uid)?;
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

    host.input_devices()
        .map_err(|error| format!("could not enumerate input devices: {error}"))?
        .find_map(|device| match device_uid(&device) {
            Ok(uid) if uid == selected_device_uid => Some(Ok(device)),
            Ok(_) => None,
            Err(error) => Some(Err(error)),
        })
        .unwrap_or_else(|| {
            Err(format!(
                "selected input device {selected_device_uid:?} is unavailable"
            ))
        })
}

fn device_config(
    host: &cpal::Host,
    selected_device_uid: Option<&str>,
) -> Result<cpal::SupportedStreamConfig, String> {
    select_input_device(host, selected_device_uid)?
        .default_input_config()
        .map_err(|error| format!("could not get default input stream configuration: {error}"))
}

fn device_uid(device: &cpal::Device) -> Result<String, String> {
    device
        .id()
        .map(|id| id.to_string())
        .map_err(|error| format!("could not read input device UID: {error}"))
}

fn spawn_pcm_worker(
    ingress: Arc<CaptureIngress>,
    telemetry: Arc<CaptureTelemetryAtomic>,
    stop_worker: Arc<AtomicBool>,
) -> JoinHandle<()> {
    thread::Builder::new()
        .name("word-covenant-pcm-meter".to_owned())
        .spawn(move || {
            while !stop_worker.load(Ordering::Acquire) {
                if !ingress.try_consume(|packet| telemetry.record_level(packet.samples)) {
                    thread::sleep(PCM_WORKER_IDLE_SLEEP);
                }
            }
            while ingress.try_consume(|_| {}) {}
        })
        .expect("PCM meter worker thread starts")
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

fn to_dbfs(value: f32) -> f32 {
    if value <= 0.0 {
        return MINIMUM_DBFS;
    }
    (20.0 * value.log10()).clamp(MINIMUM_DBFS, 0.0)
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

    #[test]
    fn converts_silence_to_a_finite_meter_floor() {
        assert_eq!(to_dbfs(0.0), MINIMUM_DBFS);
        assert_eq!(to_dbfs(1.0), 0.0);
    }

    #[test]
    fn continuous_clock_is_non_decreasing() {
        let first = continuous_time_ns();
        let second = continuous_time_ns();
        assert!(second >= first);
    }
}
