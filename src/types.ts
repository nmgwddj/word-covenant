export type SessionState = 'recording' | 'stopped'
export type TranscriptSource = 'synthetic' | 'local_inference' | 'user_edited'
export type ActionStatus = 'ready' | 'blocked' | 'completed'

export interface PrivacyStatus {
  localOnly: boolean
  egressEnabled: boolean
  activeEgressApprovals: number
  recordingSessionId: string | null
}

export interface CaptureSession {
  id: string
  startedAt: string
  state: SessionState
}

export interface TranscriptSpan {
  id: string
  sessionId: string
  captureStartNs: number
  captureEndNs: number
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
