use serde::{Deserialize, Serialize};
use std::sync::mpsc::{SyncSender, TrySendError};

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
}
