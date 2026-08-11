export type SessionState = 'starting' | 'recording' | 'stopped'
export type TranscriptSource = 'synthetic' | 'local_inference' | 'user_edited'
export type ActionStatus = 'ready' | 'blocked' | 'completed'
export type CaptureInputKind = 'microphone' | 'development_mock'
export type LocalModelKind = 'speech_recognition' | 'voice_activity_detection' | 'speaker_embedding'
export type CaptureStatus = 'idle' | 'awaiting_permission' | 'recording' | 'interrupted' | 'failed'
export type MicrophonePermission = 'not_determined' | 'granted' | 'denied' | 'restricted'
export type CaptureBridgeStatus = 'parked' | 'armed' | 'closing' | 'drained'
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

/**
 * Local-only speech gate preferences. The native runtime snapshots these
 * settings when a microphone session begins, so an active recording cannot
 * change its detection threshold midway through a session.
 */
export interface SpeechDetectionSettings {
  rmsThresholdDbfs: number
}

export interface CaptureIssue {
  code: CaptureIssueCode
  deviceName: string | null
}

/**
 * Bounded native inference bridge telemetry. It deliberately excludes PCM,
 * transcript text, model identifiers, and durable outcome payloads.
 */
export interface CaptureBridgeMetrics {
  ingressPacketsConsumed: number
  ingressDiscontinuities: number
  segmenterFailures: number
  jobsAdmitted: number
  jobsCompleted: number
  jobQueueSaturated: number
  resultQueueSaturated: number
  unavailableEngineOutcomes: number
  engineFailureOutcomes: number
  shutdownOutcomes: number
  outcomeClaimsAborted: number
  jobQueueHighWatermark: number
  resultQueueHighWatermark: number
  pendingEventHighWatermark: number
  jobQueueDepth: number
  resultQueueDepth: number
  pendingEventDepth: number
  workerHoldsOutcome: boolean
  ownedOutcomeLeaseActive: boolean
  closing: boolean
}

export interface CaptureBridgeProjection {
  status: CaptureBridgeStatus
  armed: boolean
  shutdownRequested: boolean
  workerFinished: boolean
  metrics: CaptureBridgeMetrics
}

export interface CaptureProjection {
  revision: number
  status: CaptureStatus
  permission: MicrophonePermission
  selectedDevice: CaptureInputDevice | null
  devices: CaptureInputDevice[]
  meter: CaptureMeter | null
  // Optional while older native clients are upgraded; current backends return null when absent.
  bridge?: CaptureBridgeProjection | null
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

/**
 * A user-visible, per-app-run selection of an already imported local ASR
 * model. The file path and artifact bytes remain native-only.
 */
export interface ActiveLocalAsrProfile {
  modelId: string
}

/**
 * Compact availability of the release-bundled local ASR model. Native code
 * owns resource paths and artifact verification; the WebView receives only
 * whether the bundled default can be selected for this app run.
 */
export interface BundledAsrStatus {
  available: boolean
  modelId: string | null
  message: string | null
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

/**
 * A native final transcript is available locally. The event deliberately
 * carries only an opaque session reference and sequence number; the client
 * reloads the timeline through its existing command to access transcript text.
 */
export interface FinalTranscriptProjection {
  sessionId: string
  revision: number
}

/**
 * A local, session-scoped speaking-part catalog entry with durable labels and
 * revision metadata.
 */
export interface SpeakerCluster {
  id: string
  sessionId: string
  label: string
  isUserNamed: boolean
  labelRevision: number
  aliasRevision: number
  mergedIntoClusterId: string | null
  canonicalClusterId: string
  spanCount: number
}

export interface SpeakerSpanRef {
  id: string
  revision: number
}

export interface SpeakerOperationResult {
  clusters: SpeakerCluster[]
  updatedSpans: SpeakerSpanRef[]
}

export interface AgentAction {
  id: string
  title: string
  detail: string
  status: ActionStatus
  kind: 'local_speech' | 'http_profile'
}
