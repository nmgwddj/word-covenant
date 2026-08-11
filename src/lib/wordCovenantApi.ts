import { invoke } from '@tauri-apps/api/core'
import { listen, type UnlistenFn } from '@tauri-apps/api/event'
import type {
  ActiveLocalAsrProfile,
  AgentAction,
  BundledAsrStatus,
  CaptureProjection,
  CaptureSession,
  DevelopmentMockProgress,
  FinalTranscriptProjection,
  LocalModelImportInput,
  PrivacyStatus,
  RegisteredModel,
  SpeechDetectionSettings,
  SpeakerCluster,
  SpeakerOperationResult,
  TranscriptSpan,
} from '@/types'

const demoSessionId = 'local-demo-session'
const developmentMockSessionId = 'local-development-mock-session'
const developmentMockTickNs = 200_000_000
const developmentMockTotalNs = 12_000_000_000
let demoSession: CaptureSession | null = null
let demoActions: AgentAction[] = []
let demoEgressEnabled = false
let browserActiveLocalAsrProfile: ActiveLocalAsrProfile | null = null
let browserSpeechDetectionSettings: SpeechDetectionSettings = {
  mode: 'adaptive',
  rmsThresholdDbfs: -10,
}
const browserBundledAsrStatus: BundledAsrStatus = {
  available: false,
  modelId: null,
  message: '浏览器预览不包含内置本地转写模型',
}
const browserSpeakerClustersBySession = new Map<string, SpeakerCluster[]>()

const browserCaptureProjection: CaptureProjection = {
  revision: 0,
  status: 'idle',
  permission: 'not_determined',
  selectedDevice: null,
  devices: [],
  meter: null,
  bridge: null,
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

const demoSpeakerClusters: SpeakerCluster[] = [
  {
    id: 'speaker-1',
    sessionId: demoSessionId,
    label: '说话人 1',
    isUserNamed: false,
    labelRevision: 1,
    aliasRevision: 0,
    mergedIntoClusterId: null,
    canonicalClusterId: 'speaker-1',
    spanCount: 2,
  },
  {
    id: 'speaker-2',
    sessionId: demoSessionId,
    label: '说话人 2',
    isUserNamed: false,
    labelRevision: 1,
    aliasRevision: 0,
    mergedIntoClusterId: null,
    canonicalClusterId: 'speaker-2',
    spanCount: 1,
  },
]

browserSpeakerClustersBySession.set(demoSessionId, demoSpeakerClusters)

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

function browserTimelineForSession(sessionId: string): TranscriptSpan[] | null {
  if (browserDevelopmentMock?.session.id === sessionId) {
    return browserDevelopmentMock.timeline
  }

  if (sessionId === demoSessionId) {
    return demoTimeline
  }

  return null
}

function cloneSpeakerCatalog(sessionId: string): SpeakerCluster[] {
  return demoSpeakerClusters.map(cluster => ({
    ...cluster,
    sessionId,
  }))
}

function browserSpeakerCatalog(sessionId: string): SpeakerCluster[] | null {
  return browserSpeakerClustersBySession.get(sessionId) ?? null
}

function speakerClustersWithCounts(sessionId: string): SpeakerCluster[] {
  const catalog = browserSpeakerCatalog(sessionId)
  const timeline = browserTimelineForSession(sessionId)
  if (!catalog || !timeline) return []

  const spanCounts = new Map<string, number>()
  for (const span of timeline) {
    if (span.speakerClusterId) {
      spanCounts.set(span.speakerClusterId, (spanCounts.get(span.speakerClusterId) ?? 0) + 1)
    }
  }

  return catalog.map(cluster => ({
    ...cluster,
    spanCount: spanCounts.get(cluster.id) ?? 0,
  }))
}

function browserSpeakerOperationResult(
  sessionId: string,
  updatedSpans: SpeakerOperationResult['updatedSpans'] = []
): SpeakerOperationResult {
  return {
    clusters: speakerClustersWithCounts(sessionId),
    updatedSpans,
  }
}

function resolveBrowserSpeakerSessionId(sessionId?: string): string {
  return sessionId ?? browserDevelopmentMock?.session.id ?? demoSessionId
}

function nextBrowserSpeakerId(clusters: SpeakerCluster[]): string {
  let ordinal = clusters.length + 1
  while (clusters.some(cluster => cluster.id === `speaker-${ordinal}`)) {
    ordinal += 1
  }
  return `speaker-${ordinal}`
}

function updateBrowserTimelineSpan(
  sessionId: string,
  logicalSpanId: string,
  update: (span: TranscriptSpan) => TranscriptSpan
): TranscriptSpan {
  const timeline = browserTimelineForSession(sessionId)
  if (!timeline) {
    throw new Error('本地会话不存在')
  }

  const index = timeline.findIndex(span => span.id === logicalSpanId)
  if (index < 0) {
    throw new Error('记录片段不存在')
  }

  const updated = update(timeline[index]!)
  if (browserDevelopmentMock?.session.id === sessionId) {
    browserDevelopmentMock = {
      ...browserDevelopmentMock,
      timeline: timeline.map((span, spanIndex) => (spanIndex === index ? updated : span)),
    }
  } else {
    demoTimeline[index] = updated
  }
  return updated
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

  async getSpeechDetectionSettings(): Promise<SpeechDetectionSettings> {
    if (isTauriRuntime()) {
      return invoke<SpeechDetectionSettings>('get_speech_detection_settings')
    }

    return { ...browserSpeechDetectionSettings }
  },

  async setSpeechDetectionSettings(input: SpeechDetectionSettings): Promise<SpeechDetectionSettings> {
    if (isTauriRuntime()) {
      return invoke<SpeechDetectionSettings>('set_speech_detection_settings', { input })
    }

    browserSpeechDetectionSettings = { ...input }
    return { ...browserSpeechDetectionSettings }
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
      return listen<CaptureProjection>('capture-projection', event => listener(event.payload))
    }

    return () => {}
  },

  async onFinalTranscriptProjection(listener: (projection: FinalTranscriptProjection) => void): Promise<UnlistenFn> {
    if (isTauriRuntime()) {
      return listen<FinalTranscriptProjection>('final-transcript-projection', event => listener(event.payload))
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

    if (browserDevelopmentMock && (!sessionId || sessionId === browserDevelopmentMock.session.id)) {
      return browserDevelopmentMock.timeline
    }

    return sessionId && sessionId !== demoSessionId ? [] : demoTimeline
  },

  async listSpeakerClusters(sessionId?: string): Promise<SpeakerCluster[]> {
    if (isTauriRuntime()) {
      return invoke<SpeakerCluster[]>('list_speaker_clusters', { sessionId })
    }

    return speakerClustersWithCounts(resolveBrowserSpeakerSessionId(sessionId))
  },

  async createSpeakerCluster(input: { sessionId: string }): Promise<SpeakerOperationResult> {
    if (isTauriRuntime()) {
      return invoke<SpeakerOperationResult>('create_speaker_cluster', { input })
    }

    const catalog = browserSpeakerCatalog(input.sessionId)
    if (!catalog || !browserTimelineForSession(input.sessionId)) {
      throw new Error('本地会话不存在')
    }

    const id = nextBrowserSpeakerId(catalog)
    catalog.push({
      id,
      sessionId: input.sessionId,
      label: `说话人 ${id.replace('speaker-', '')}`,
      isUserNamed: false,
      labelRevision: 1,
      aliasRevision: 0,
      mergedIntoClusterId: null,
      canonicalClusterId: id,
      spanCount: 0,
    })
    return browserSpeakerOperationResult(input.sessionId)
  },

  async renameSpeakerCluster(input: {
    sessionId: string
    clusterId: string
    expectedLabelRevision: number
    label: string
  }): Promise<SpeakerOperationResult> {
    if (isTauriRuntime()) {
      return invoke<SpeakerOperationResult>('rename_speaker_cluster', { input })
    }

    const catalog = browserSpeakerCatalog(input.sessionId)
    const cluster = catalog?.find(item => item.id === input.clusterId)
    if (!cluster) {
      throw new Error('说话人归类不存在')
    }
    if (cluster.labelRevision !== input.expectedLabelRevision) {
      throw new Error('名称已被更新，请刷新后重试')
    }
    const label = input.label.trim()
    if (!label) {
      throw new Error('名称不能为空')
    }

    Object.assign(cluster, {
      label,
      isUserNamed: true,
      labelRevision: cluster.labelRevision + 1,
    })
    return browserSpeakerOperationResult(input.sessionId)
  },

  async reassignTranscriptSpeaker(input: {
    sessionId: string
    logicalSpanId: string
    expectedRevision: number
    targetClusterId: string | null
  }): Promise<SpeakerOperationResult> {
    if (isTauriRuntime()) {
      return invoke<SpeakerOperationResult>('reassign_transcript_speaker', { input })
    }

    if (
      input.targetClusterId &&
      !browserSpeakerCatalog(input.sessionId)?.some(
        cluster => cluster.id === input.targetClusterId && cluster.mergedIntoClusterId === null
      )
    ) {
      throw new Error('目标说话人归类不存在')
    }

    const updated = updateBrowserTimelineSpan(input.sessionId, input.logicalSpanId, span => {
      if (span.revision !== input.expectedRevision) {
        throw new Error('记录已被更新，请刷新后重试')
      }
      return {
        ...span,
        speakerClusterId: input.targetClusterId,
        revision: span.revision + 1,
      }
    })
    return browserSpeakerOperationResult(input.sessionId, [
      {
        id: updated.id,
        revision: updated.revision,
      },
    ])
  },

  async listLocalModels(): Promise<RegisteredModel[]> {
    if (isTauriRuntime()) {
      return invoke<RegisteredModel[]>('list_local_models')
    }

    return []
  },

  async getBundledAsrStatus(): Promise<BundledAsrStatus> {
    if (isTauriRuntime()) {
      return invoke<BundledAsrStatus>('get_bundled_asr_status')
    }

    return { ...browserBundledAsrStatus }
  },

  async getActiveLocalAsrProfile(): Promise<ActiveLocalAsrProfile | null> {
    if (isTauriRuntime()) {
      return invoke<ActiveLocalAsrProfile | null>('get_active_local_asr_profile')
    }

    return browserActiveLocalAsrProfile
  },

  async selectActiveLocalAsrModel(modelId: string): Promise<ActiveLocalAsrProfile> {
    if (isTauriRuntime()) {
      return invoke<ActiveLocalAsrProfile>('select_active_local_asr_model', {
        input: { modelId },
      })
    }

    throw new Error('浏览器预览不能启用本地转写模型')
  },

  async selectLocalModelFile(): Promise<string | null> {
    if (isTauriRuntime()) {
      return invoke<string | null>('select_local_model_file')
    }

    throw new Error('浏览器预览不能打开本机模型文件选择器')
  },

  async importLocalModel(input: LocalModelImportInput): Promise<RegisteredModel> {
    if (isTauriRuntime()) {
      return invoke<RegisteredModel>('import_local_model', { input })
    }

    throw new Error('浏览器预览不能导入本地模型文件')
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
    browserSpeakerClustersBySession.set(session.id, cloneSpeakerCatalog(session.id))
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
      browserDevelopmentMock.elapsedNs + developmentMockTickNs
    )
    const newSpans: TranscriptSpan[] = []
    while (
      browserDevelopmentMock.nextCueIndex < developmentMockCues.length &&
      developmentMockCues[browserDevelopmentMock.nextCueIndex]!.captureEndNs <= browserDevelopmentMock.elapsedNs
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
