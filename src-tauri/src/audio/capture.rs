use crossbeam_queue::ArrayQueue;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{SyncSender, TrySendError};
use std::sync::Arc;

/// The ingress has a fixed upper bound so an unavailable downstream stage can
/// never turn microphone input into unbounded memory growth.
pub const CAPTURE_INGRESS_CAPACITY: usize = 96;
pub const MAX_CAPTURE_SAMPLES_PER_PACKET: usize = 8_192;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CapturePacket {
    pub starting_sample_offset: u64,
    pub sample_rate: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
}

impl CapturePacket {
    pub fn frame_count(&self) -> usize {
        let channels = usize::from(self.channels);
        if channels == 0 {
            return 0;
        }
        self.samples.len() / channels
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CaptureWriteResult {
    Enqueued,
    Dropped,
    Closed,
}

pub struct BoundedCaptureWriter {
    sender: SyncSender<CapturePacket>,
}

/// Pre-allocated PCM ingress for a native realtime callback.
///
/// A callback takes an empty slot, fills it, and attempts to publish it. The
/// consumer returns the slot after calculating non-audio telemetry. Neither
/// path allocates, blocks, calls Tauri, or touches application state.
pub struct CaptureIngress {
    available: ArrayQueue<CaptureSlot>,
    ready: ArrayQueue<CaptureSlot>,
    dropped_packets: AtomicU64,
}

#[derive(Debug)]
struct CaptureSlot {
    starting_sample_offset: u64,
    sample_rate: u32,
    channels: u16,
    len: usize,
    samples: Box<[f32]>,
}

#[derive(Clone, Copy, Debug)]
pub struct CaptureIngressPacket<'a> {
    pub starting_sample_offset: u64,
    pub sample_rate: u32,
    pub channels: u16,
    pub samples: &'a [f32],
}

impl CaptureIngress {
    pub fn new(capacity: usize, maximum_samples_per_packet: usize) -> Result<Arc<Self>, String> {
        if capacity == 0 {
            return Err("capture ingress capacity must be greater than zero".to_owned());
        }
        if maximum_samples_per_packet == 0 {
            return Err("capture ingress packet size must be greater than zero".to_owned());
        }

        let available = ArrayQueue::new(capacity);
        let ready = ArrayQueue::new(capacity);
        for _ in 0..capacity {
            available
                .push(CaptureSlot {
                    starting_sample_offset: 0,
                    sample_rate: 0,
                    channels: 0,
                    len: 0,
                    samples: vec![0.0; maximum_samples_per_packet].into_boxed_slice(),
                })
                .expect("empty pre-allocated capture queue accepts its slots");
        }

        Ok(Arc::new(Self {
            available,
            ready,
            dropped_packets: AtomicU64::new(0),
        }))
    }

    pub fn default_sized() -> Arc<Self> {
        Self::new(CAPTURE_INGRESS_CAPACITY, MAX_CAPTURE_SAMPLES_PER_PACKET)
            .expect("capture ingress constants are valid")
    }

    pub fn dropped_packets(&self) -> u64 {
        self.dropped_packets.load(Ordering::Relaxed)
    }

    /// Returns a point-in-time count of queued packets without exposing PCM.
    ///
    /// The value is intended for bounded native maintenance paths. A callback
    /// may enqueue another packet immediately after this snapshot.
    pub(crate) fn queued_packet_count(&self) -> usize {
        self.ready.len()
    }

    /// Copy a normalized PCM buffer into a free fixed-capacity slot.
    pub fn try_write(
        &self,
        starting_sample_offset: u64,
        sample_rate: u32,
        channels: u16,
        samples: &[f32],
    ) -> CaptureWriteResult {
        self.try_write_mapped(
            starting_sample_offset,
            sample_rate,
            channels,
            samples,
            |sample| sample,
        )
    }

    /// Convert a native sample slice directly into a pre-allocated slot.
    /// This is used by the CPAL callback to avoid constructing a temporary
    /// `Vec<f32>` on the audio thread.
    pub fn try_write_mapped<T: Copy>(
        &self,
        starting_sample_offset: u64,
        sample_rate: u32,
        channels: u16,
        samples: &[T],
        map_sample: impl Fn(T) -> f32,
    ) -> CaptureWriteResult {
        if sample_rate == 0 || channels == 0 || !samples.len().is_multiple_of(usize::from(channels))
        {
            self.record_drop();
            return CaptureWriteResult::Dropped;
        }

        let Some(mut slot) = self.available.pop() else {
            self.record_drop();
            return CaptureWriteResult::Dropped;
        };
        if samples.len() > slot.samples.len() {
            self.return_unconsumed_slot(slot);
            self.record_drop();
            return CaptureWriteResult::Dropped;
        }

        slot.starting_sample_offset = starting_sample_offset;
        slot.sample_rate = sample_rate;
        slot.channels = channels;
        slot.len = samples.len();
        for (destination, source) in slot.samples[..samples.len()]
            .iter_mut()
            .zip(samples.iter().copied())
        {
            *destination = map_sample(source);
        }

        match self.ready.push(slot) {
            Ok(()) => CaptureWriteResult::Enqueued,
            Err(slot) => {
                self.return_unconsumed_slot(slot);
                self.record_drop();
                CaptureWriteResult::Dropped
            }
        }
    }

    /// Process one queued packet and recycle its slot before returning.
    pub fn try_consume(&self, consume: impl FnOnce(CaptureIngressPacket<'_>)) -> bool {
        let Some(slot) = self.ready.pop() else {
            return false;
        };
        consume(CaptureIngressPacket {
            starting_sample_offset: slot.starting_sample_offset,
            sample_rate: slot.sample_rate,
            channels: slot.channels,
            samples: &slot.samples[..slot.len],
        });
        self.return_consumed_slot(slot);
        true
    }

    fn record_drop(&self) {
        self.dropped_packets.fetch_add(1, Ordering::Relaxed);
    }

    fn return_unconsumed_slot(&self, mut slot: CaptureSlot) {
        slot.len = 0;
        if self.available.push(slot).is_err() {
            debug_assert!(false, "capture ingress slot was returned twice");
        }
    }

    fn return_consumed_slot(&self, mut slot: CaptureSlot) {
        slot.samples[..slot.len].fill(0.0);
        self.return_unconsumed_slot(slot);
    }
}

impl BoundedCaptureWriter {
    pub fn new(sender: SyncSender<CapturePacket>) -> Self {
        Self { sender }
    }

    pub fn try_write(&self, packet: CapturePacket) -> CaptureWriteResult {
        match self.sender.try_send(packet) {
            Ok(()) => CaptureWriteResult::Enqueued,
            Err(TrySendError::Full(_)) => CaptureWriteResult::Dropped,
            Err(TrySendError::Disconnected(_)) => CaptureWriteResult::Closed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::sync_channel;

    fn packet() -> CapturePacket {
        CapturePacket {
            starting_sample_offset: 0,
            sample_rate: 16_000,
            channels: 1,
            samples: vec![0.0; 160],
        }
    }

    #[test]
    fn reports_dropped_packets_when_the_bounded_queue_is_full() {
        let (sender, _receiver) = sync_channel(1);
        let writer = BoundedCaptureWriter::new(sender);

        assert_eq!(writer.try_write(packet()), CaptureWriteResult::Enqueued);
        assert_eq!(writer.try_write(packet()), CaptureWriteResult::Dropped);
    }

    #[test]
    fn calculates_interleaved_frame_count() {
        let mut stereo_packet = packet();
        stereo_packet.channels = 2;
        stereo_packet.samples = vec![0.0; 320];

        assert_eq!(stereo_packet.frame_count(), 160);
    }

    #[test]
    fn preallocated_ingress_drops_when_no_slot_is_available() {
        let ingress = CaptureIngress::new(1, 4).unwrap();

        assert_eq!(
            ingress.try_write(0, 16_000, 1, &[0.1, 0.2]),
            CaptureWriteResult::Enqueued
        );
        assert_eq!(
            ingress.try_write(2, 16_000, 1, &[0.3, 0.4]),
            CaptureWriteResult::Dropped
        );
        assert_eq!(ingress.dropped_packets(), 1);
    }

    #[test]
    fn ingress_recycles_a_slot_after_consuming_pcm() {
        let ingress = CaptureIngress::new(1, 4).unwrap();
        ingress.try_write(8, 16_000, 1, &[0.1, 0.2]);

        let mut observed = Vec::new();
        assert!(ingress.try_consume(|packet| {
            observed = vec![
                packet.starting_sample_offset as f32,
                packet.sample_rate as f32,
                packet.channels as f32,
                packet.samples[0],
                packet.samples[1],
            ];
        }));
        assert_eq!(observed, vec![8.0, 16_000.0, 1.0, 0.1, 0.2]);
        assert_eq!(
            ingress.try_write(10, 16_000, 1, &[0.3, 0.4]),
            CaptureWriteResult::Enqueued
        );
    }

    #[test]
    fn ingress_clears_consumed_pcm_before_recycling_its_slot() {
        let ingress = CaptureIngress::new(1, 4).unwrap();
        assert_eq!(
            ingress.try_write(8, 16_000, 1, &[0.1, -0.2, 0.3]),
            CaptureWriteResult::Enqueued
        );

        assert!(ingress.try_consume(|packet| {
            assert_eq!(packet.samples, &[0.1, -0.2, 0.3]);
        }));

        let recycled = ingress
            .available
            .pop()
            .expect("consuming a packet returns its slot to the available queue");
        assert_eq!(recycled.len, 0);
        assert_eq!(&recycled.samples[..3], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn ingress_rejects_malformed_or_oversized_pcm_without_retaining_it() {
        let ingress = CaptureIngress::new(1, 2).unwrap();

        assert_eq!(
            ingress.try_write(0, 16_000, 2, &[0.1]),
            CaptureWriteResult::Dropped
        );
        assert_eq!(
            ingress.try_write(0, 16_000, 1, &[0.1, 0.2, 0.3]),
            CaptureWriteResult::Dropped
        );
        assert_eq!(ingress.dropped_packets(), 2);
    }
}
