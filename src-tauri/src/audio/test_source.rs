use super::CapturePacket;
use std::collections::VecDeque;

pub trait CaptureSource {
    fn start(&mut self) -> Result<(), String>;
    fn stop(&mut self) -> Result<(), String>;
    fn try_next_packet(&mut self) -> Option<CapturePacket>;
}

pub struct TestCaptureSource {
    started: bool,
    packets: VecDeque<CapturePacket>,
}

impl TestCaptureSource {
    pub fn new(packets: impl IntoIterator<Item = CapturePacket>) -> Self {
        Self {
            started: false,
            packets: packets.into_iter().collect(),
        }
    }
}

impl CaptureSource for TestCaptureSource {
    fn start(&mut self) -> Result<(), String> {
        self.started = true;
        Ok(())
    }

    fn stop(&mut self) -> Result<(), String> {
        self.started = false;
        Ok(())
    }

    fn try_next_packet(&mut self) -> Option<CapturePacket> {
        self.started.then(|| self.packets.pop_front()).flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_emits_test_packets_while_started() {
        let packet = CapturePacket {
            starting_sample_offset: 0,
            sample_rate: 16_000,
            channels: 1,
            samples: vec![0.0; 16],
        };
        let mut source = TestCaptureSource::new([packet.clone()]);

        assert_eq!(source.try_next_packet(), None);
        source.start().unwrap();
        assert_eq!(source.try_next_packet(), Some(packet));
        source.stop().unwrap();
        assert_eq!(source.try_next_packet(), None);
    }
}
