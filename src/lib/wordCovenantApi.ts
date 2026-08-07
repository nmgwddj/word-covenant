import { invoke } from '@tauri-apps/api/core'
import type { AgentAction, CaptureSession, PrivacyStatus, TranscriptSpan } from '@/types'

const demoSessionId = 'local-demo-session'
let demoSession: CaptureSession | null = null
let demoActions: AgentAction[] = []
let demoEgressEnabled = false

const demoTimeline: TranscriptSpan[] = [
  {
    id: 'span-001',
    sessionId: demoSessionId,
    captureStartNs: 0,
    captureEndNs: 2_800_000_000,
    speakerClusterId: 'speaker-1',
    text: '本次记录仅保存在本机。',
    isFinal: true,
    revision: 1,
    source: 'synthetic',
  },
  {
    id: 'span-002',
    sessionId: demoSessionId,
    captureStartNs: 3_100_000_000,
    captureEndNs: 7_200_000_000,
    speakerClusterId: 'speaker-2',
    text: '出网行为需要在行动前单独授权。',
    isFinal: true,
    revision: 1,
    source: 'synthetic',
  },
  {
    id: 'span-003',
    sessionId: demoSessionId,
    captureStartNs: 7_600_000_000,
    captureEndNs: 11_400_000_000,
    speakerClusterId: 'speaker-1',
    text: '先生成一份待确认的行动草案。',
    isFinal: true,
    revision: 1,
    source: 'synthetic',
  },
]

function isTauriRuntime(): boolean {
  return typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window
}

function createDemoSession(): CaptureSession {
  return {
    id: demoSessionId,
    startedAt: new Date().toISOString(),
    state: 'recording',
  }
}

function getDemoPrivacyStatus(): PrivacyStatus {
  return {
    // Browser preview models the policy state only; it never sends HTTP requests.
    localOnly: !demoEgressEnabled,
    egressEnabled: demoEgressEnabled,
    activeEgressApprovals: 0,
    recordingSessionId: demoSession?.state === 'recording' ? demoSession.id : null,
  }
}

export const wordCovenantApi = {
  async getPrivacyStatus(): Promise<PrivacyStatus> {
    if (isTauriRuntime()) {
      return invoke<PrivacyStatus>('get_privacy_status')
    }

    return getDemoPrivacyStatus()
  },

  async setEgressEnabled(enabled: boolean): Promise<PrivacyStatus> {
    if (isTauriRuntime()) {
      return invoke<PrivacyStatus>('set_egress_enabled', { enabled })
    }

    demoEgressEnabled = enabled
    return getDemoPrivacyStatus()
  },

  async startSession(): Promise<CaptureSession> {
    if (isTauriRuntime()) {
      return invoke<CaptureSession>('start_session')
    }

    demoSession = createDemoSession()
    return demoSession
  },

  async stopSession(): Promise<CaptureSession | null> {
    if (isTauriRuntime()) {
      return invoke<CaptureSession | null>('stop_session')
    }

    if (demoSession) {
      demoSession = { ...demoSession, state: 'stopped' }
    }
    return demoSession
  },

  async listTimeline(sessionId?: string): Promise<TranscriptSpan[]> {
    if (isTauriRuntime()) {
      return invoke<TranscriptSpan[]>('list_timeline', { sessionId })
    }

    return sessionId && sessionId !== demoSessionId ? [] : demoTimeline
  },

  async proposeLocalSpeech(): Promise<AgentAction> {
    if (isTauriRuntime()) {
      return invoke<AgentAction>('propose_local_speech')
    }

    const action: AgentAction = {
      id: `action-${demoActions.length + 1}`,
      title: '播报本地行动摘要',
      detail: '仅使用本机语音输出',
      status: 'ready',
      kind: 'local_speech',
    }
    demoActions = [action, ...demoActions]
    return action
  },

  async listActions(): Promise<AgentAction[]> {
    if (isTauriRuntime()) {
      return invoke<AgentAction[]>('list_actions')
    }

    return demoActions
  },
}
