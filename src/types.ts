export type SessionState = 'recording' | 'stopped'
export type TranscriptSource = 'synthetic' | 'local_inference' | 'user_edited'
export type ActionStatus = 'ready' | 'blocked' | 'completed'
export type CaptureInputKind = 'microphone' | 'development_mock'
export type LocalModelKind = 'speech_recognition' | 'voice_activity_detection' | 'speaker_embedding'
export type CaptureStatus = 'idle' | 'awaiting_permission' | 'recording' | 'interrupted' | 'failed'
export type MicrophonePermission = 'not_determined' | 'granted' | 'denied' | 'restricted'
export type CaptureIssueCode =
  | 'permission_denied'
  | 'permission_restricted'
  | 'no_input_device'
  | 'input_device_unavailable'
  | 'stream_start_failed'
  | 'capture_queue_overrun'
  | 'capture_queue_closed'

export interface PrivacyStatus {
  localOnly: boolean
  egressEnabled: boolean
  activeEgressApprovals: number
  recordingSessionId: string | null
}

export interface CaptureSession {
  id: string
  startedAt: string
  startedMonotonicNs: number
  stoppedAt: string | null
  state: SessionState
}

export interface CaptureInputDevice {
  uid: string
  name: string
}

export interface CaptureMeter {
  rmsDbfs: number
  peakDbfs: number
  clipping: boolean
  droppedPackets: number
}

export interface CaptureIssue {
  code: CaptureIssueCode
  deviceName: string | null
}

export interface CaptureProjection {
  revision: number
  status: CaptureStatus
  permission: MicrophonePermission
  selectedDevice: CaptureInputDevice | null
  devices: CaptureInputDevice[]
  meter: CaptureMeter | null
  lastIssue: CaptureIssue | null
}

export interface DevelopmentMockProgress {
  sessionId: string
  packetsAdvanced: number
  spans: TranscriptSpan[]
  exhausted: boolean
}

export interface RegisteredModel {
  id: string
  modelKind: LocalModelKind
  fileSizeBytes: number
  sha256: string
  version: string
  inputFormat: string
  modelCardId: string
  licenseId: string
  licenseConfirmedAt: string
  importedAt: string
}

export interface LocalModelImportInput {
  sourcePath: string
  modelKind: LocalModelKind
  version: string
  inputFormat: string
  expectedSha256: string
  modelCardId: string
  licenseId: string
  licenseAcknowledged: boolean
}

export interface TranscriptSpan {
  id: string
  sessionId: string
  captureStartNs: number
  captureEndNs: number
  wallClockStart?: string | null
  speakerClusterId: string | null
  text: string
  isFinal: boolean
  revision: number
  source: TranscriptSource
}

export interface AgentAction {
  id: string
  title: string
  detail: string
  status: ActionStatus
  kind: 'local_speech' | 'http_profile'
}
