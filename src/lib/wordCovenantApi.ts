import { invoke } from '@tauri-apps/api/core'
import { listen, type UnlistenFn } from '@tauri-apps/api/event'
import type {
  AgentAction,
  CaptureProjection,
  CaptureSession,
  DevelopmentMockProgress,
  PrivacyStatus,
  TranscriptSpan,
} from '@/types'

const demoSessionId = 'local-demo-session'
const developmentMockSessionId = 'local-development-mock-session'
const developmentMockTickNs = 200_000_000
const developmentMockTotalNs = 12_000_000_000
let demoSession: CaptureSession | null = null
let demoActions: AgentAction[] = []
let demoEgressEnabled = false

const browserCaptureProjection: CaptureProjection = {
  revision: 0,
  status: 'idle',
  permission: 'not_determined',
  selectedDevice: null,
  devices: [],
  meter: null,
  lastIssue: null,
}

interface BrowserDevelopmentMock {
  active: boolean
  elapsedNs: number
  nextCueIndex: number
  session: CaptureSession
  timeline: TranscriptSpan[]
}

let browserDevelopmentMock: BrowserDevelopmentMock | null = null

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

const developmentMockCues = [
  {
    id: 'development-mock-span-001',
    captureStartNs: 0,
    captureEndNs: 2_800_000_000,
    speakerClusterId: 'speaker-1',
    text: '本次记录仅保存在本机。',
    isFinal: true,
    revision: 1,
    source: 'synthetic' as const,
  },
  {
    id: 'development-mock-span-002',
    captureStartNs: 3_100_000_000,
    captureEndNs: 7_200_000_000,
    speakerClusterId: 'speaker-2',
    text: '出网行为需要在行动前单独授权。',
    isFinal: true,
    revision: 1,
    source: 'synthetic' as const,
  },
  {
    id: 'development-mock-span-003',
    captureStartNs: 7_600_000_000,
    captureEndNs: 11_400_000_000,
    speakerClusterId: 'speaker-1',
    text: '先生成一份待确认的行动草案。',
    isFinal: true,
    revision: 1,
    source: 'synthetic' as const,
  },
]

function isTauriRuntime(): boolean {
  return typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window
}

function createDemoSession(id = demoSessionId): CaptureSession {
  return {
    id,
    startedAt: new Date().toISOString(),
    startedMonotonicNs: 0,
    stoppedAt: null,
    state: 'recording',
  }
}

function browserMockProgress(newSpans: TranscriptSpan[] = []): DevelopmentMockProgress {
  if (!browserDevelopmentMock) {
    throw new Error('development mock capture is not active')
  }

  return {
    sessionId: browserDevelopmentMock.session.id,
    packetsAdvanced: 10,
    spans: newSpans,
    exhausted: !browserDevelopmentMock.active,
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

    throw new Error('浏览器预览不提供真实麦克风输入；请选择开发模拟音频输入')
  },

  async getCaptureProjection(): Promise<CaptureProjection> {
    if (isTauriRuntime()) {
      return invoke<CaptureProjection>('get_capture_projection')
    }

    return browserCaptureProjection
  },

  async selectInputDevice(deviceUid: string): Promise<CaptureProjection> {
    if (isTauriRuntime()) {
      return invoke<CaptureProjection>('select_input_device', { input: { deviceUid } })
    }

    throw new Error('浏览器预览不提供真实麦克风选择')
  },

  async onCaptureProjection(listener: (projection: CaptureProjection) => void): Promise<UnlistenFn> {
    if (isTauriRuntime()) {
      return listen<CaptureProjection>('capture-projection', (event) => listener(event.payload))
    }

    return () => {}
  },

  async stopSession(): Promise<CaptureSession | null> {
    if (isTauriRuntime()) {
      return invoke<CaptureSession | null>('stop_session')
    }

    if (demoSession) {
      demoSession = { ...demoSession, state: 'stopped', stoppedAt: new Date().toISOString() }
      if (browserDevelopmentMock?.session.id === demoSession.id) {
        browserDevelopmentMock = {
          ...browserDevelopmentMock,
          active: false,
          session: demoSession,
        }
      }
    }
    return demoSession
  },

  async listTimeline(sessionId?: string): Promise<TranscriptSpan[]> {
    if (isTauriRuntime()) {
      return invoke<TranscriptSpan[]>('list_timeline', { sessionId })
    }

    if (
      browserDevelopmentMock
      && (!sessionId || sessionId === browserDevelopmentMock.session.id)
    ) {
      return browserDevelopmentMock.timeline
    }

    return sessionId && sessionId !== demoSessionId ? [] : demoTimeline
  },

  async startDevelopmentMockSession(): Promise<CaptureSession> {
    if (isTauriRuntime()) {
      return invoke<CaptureSession>('start_development_mock_session')
    }

    if (demoSession?.state === 'recording') {
      throw new Error('a capture session is already recording')
    }

    const session = createDemoSession(developmentMockSessionId)
    demoSession = session
    browserDevelopmentMock = {
      active: true,
      elapsedNs: 0,
      nextCueIndex: 0,
      session,
      timeline: [],
    }
    return session
  },

  async advanceDevelopmentMock(): Promise<DevelopmentMockProgress> {
    if (isTauriRuntime()) {
      return invoke<DevelopmentMockProgress>('advance_development_mock', {
        input: { packetCount: 10 },
      })
    }

    if (!browserDevelopmentMock?.active) {
      throw new Error('development mock capture is not active')
    }

    browserDevelopmentMock.elapsedNs = Math.min(
      developmentMockTotalNs,
      browserDevelopmentMock.elapsedNs + developmentMockTickNs,
    )
    const newSpans: TranscriptSpan[] = []
    while (
      browserDevelopmentMock.nextCueIndex < developmentMockCues.length
      && developmentMockCues[browserDevelopmentMock.nextCueIndex]!.captureEndNs
        <= browserDevelopmentMock.elapsedNs
    ) {
      const cue = developmentMockCues[browserDevelopmentMock.nextCueIndex]!
      const span: TranscriptSpan = {
        ...cue,
        captureStartNs: cue.captureStartNs + browserDevelopmentMock.session.startedMonotonicNs,
        captureEndNs: cue.captureEndNs + browserDevelopmentMock.session.startedMonotonicNs,
        sessionId: browserDevelopmentMock.session.id,
      }
      browserDevelopmentMock.timeline = [...browserDevelopmentMock.timeline, span]
      newSpans.push(span)
      browserDevelopmentMock.nextCueIndex += 1
    }

    if (browserDevelopmentMock.elapsedNs === developmentMockTotalNs) {
      const stoppedSession: CaptureSession = {
        ...browserDevelopmentMock.session,
        state: 'stopped',
        stoppedAt: new Date().toISOString(),
      }
      browserDevelopmentMock = {
        ...browserDevelopmentMock,
        active: false,
        session: stoppedSession,
      }
      demoSession = stoppedSession
    }

    return browserMockProgress(newSpans)
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
