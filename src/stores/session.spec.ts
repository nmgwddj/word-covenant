import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, test, vi } from 'vitest'
import { wordCovenantApi } from '@/lib/wordCovenantApi'
import { useModelStore } from '@/stores/models'
import { useSessionStore } from './session'

const recordingSession = {
  id: 'development-session',
  startedAt: '2026-08-08T00:00:00.000Z',
  startedMonotonicNs: 1_000,
  stoppedAt: null,
  state: 'recording' as const,
}

const olderSummary = {
  id: 'session-older',
  startedAt: '2026-08-07T00:00:00.000Z',
  startedMonotonicNs: 100,
  stoppedAt: '2026-08-07T00:00:10.000Z',
  state: 'stopped' as const,
  transcriptCount: 1,
}

const newestSummary = {
  id: 'session-newest',
  startedAt: '2026-08-08T00:00:00.000Z',
  startedMonotonicNs: 1_000,
  stoppedAt: '2026-08-08T00:00:12.000Z',
  state: 'stopped' as const,
  transcriptCount: 2,
}

const newestTimeline = [
  {
    id: 'newest-span',
    sessionId: newestSummary.id,
    captureStartNs: 1_100,
    captureEndNs: 1_900,
    speakerClusterId: null,
    text: '最新会话',
    isFinal: true,
    revision: 1,
    source: 'local_inference' as const,
  },
]

const newestClusters = [
  {
    id: 'speaker-1',
    sessionId: newestSummary.id,
    label: '说话人 1',
    isUserNamed: false,
    labelRevision: 1,
    aliasRevision: 0,
    mergedIntoClusterId: null,
    canonicalClusterId: 'speaker-1',
    spanCount: 1,
    canEnrollVoiceProfile: true,
  },
]

describe('session store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.restoreAllMocks()
  })

  test('initializes the newest session archive projection by explicit session id', async () => {
    const listSessions = vi.spyOn(wordCovenantApi, 'listSessions').mockResolvedValue([newestSummary, olderSummary])
    const listTimeline = vi.spyOn(wordCovenantApi, 'listTimeline').mockResolvedValue(newestTimeline)
    const listSpeakerClusters = vi.spyOn(wordCovenantApi, 'listSpeakerClusters').mockResolvedValue(newestClusters)
    vi.spyOn(wordCovenantApi, 'listActions').mockResolvedValue([])
    vi.spyOn(wordCovenantApi, 'getCaptureProjection').mockResolvedValue({
      revision: 0,
      status: 'idle',
      permission: 'granted',
      selectedDevice: null,
      devices: [],
      meter: null,
      lastIssue: null,
    })
    const store = useSessionStore()

    await store.initialize()

    expect(listSessions).toHaveBeenCalledOnce()
    expect(listTimeline).toHaveBeenCalledWith(newestSummary.id)
    expect(listSpeakerClusters).toHaveBeenCalledWith(newestSummary.id)
    expect(store.sessions).toEqual([newestSummary, olderSummary])
    expect(store.selectedSessionId).toBe(newestSummary.id)
    expect(store.selectedSession).toEqual(newestSummary)
    expect(store.timeline).toEqual(newestTimeline)
    expect(store.speakerClusters).toEqual(newestClusters)
    expect(store.isSessionHistoryLoading).toBe(false)
    expect(store.sessionHistoryError).toBeNull()
  })

  test('selects a newly started recording and refreshes its archive summary', async () => {
    const startedSummary = { ...recordingSession, transcriptCount: 0 }
    const store = useSessionStore()
    vi.spyOn(wordCovenantApi, 'startSession').mockResolvedValue(recordingSession)
    vi.spyOn(wordCovenantApi, 'listTimeline').mockResolvedValue([])
    vi.spyOn(wordCovenantApi, 'listSpeakerClusters').mockResolvedValue([])
    vi.spyOn(wordCovenantApi, 'getCaptureProjection').mockResolvedValue({
      revision: 1,
      status: 'recording',
      permission: 'granted',
      selectedDevice: null,
      devices: [],
      meter: null,
      lastIssue: null,
    })
    const listSessions = vi.spyOn(wordCovenantApi, 'listSessions').mockResolvedValue([startedSummary, newestSummary])

    await store.toggleRecording()

    expect(store.activeSession).toEqual(recordingSession)
    expect(store.selectedSessionId).toBe(recordingSession.id)
    expect(store.selectedSession).toEqual(startedSummary)
    expect(store.timeline).toEqual([])
    expect(listSessions).toHaveBeenCalledOnce()
  })

  test('stops into a final timeline and reports a failed archive refresh without discarding it', async () => {
    const stoppedSession = {
      ...recordingSession,
      state: 'stopped' as const,
      stoppedAt: '2026-08-08T00:00:12.000Z',
    }
    const finalTimeline = [{ ...newestTimeline[0], sessionId: stoppedSession.id }]
    const store = useSessionStore()
    store.activeSession = recordingSession
    store.selectedSessionId = recordingSession.id
    store.sessions = [{ ...recordingSession, transcriptCount: 0 }]
    store.applyCaptureProjection({
      revision: 1,
      status: 'recording',
      permission: 'granted',
      selectedDevice: null,
      devices: [],
      meter: null,
      lastIssue: null,
    })
    vi.spyOn(wordCovenantApi, 'stopSession').mockResolvedValue(stoppedSession)
    vi.spyOn(wordCovenantApi, 'listTimeline').mockResolvedValue(finalTimeline)
    vi.spyOn(wordCovenantApi, 'listSessions').mockRejectedValue(new Error('会话索引暂不可用'))

    await store.toggleRecording()

    expect(store.activeSession).toEqual(stoppedSession)
    expect(store.selectedSessionId).toBe(stoppedSession.id)
    expect(store.timeline).toEqual(finalTimeline)
    expect(store.sessions).toEqual([{ ...stoppedSession, transcriptCount: 1 }])
    expect(store.sessionHistoryError).toBe('会话索引暂不可用')
    expect(store.isSessionHistoryLoading).toBe(false)
  })

  test('retains session summaries when the newest timeline cannot be loaded during initialization', async () => {
    const store = useSessionStore()
    vi.spyOn(wordCovenantApi, 'listSessions').mockResolvedValue([newestSummary, olderSummary])
    vi.spyOn(wordCovenantApi, 'listTimeline').mockRejectedValue(new Error('时间线暂不可用'))
    vi.spyOn(wordCovenantApi, 'listSpeakerClusters').mockResolvedValue([])

    await store.initializeSessionHistory()

    expect(store.sessions).toEqual([newestSummary, olderSummary])
    expect(store.selectedSessionId).toBeNull()
    expect(store.timeline).toEqual([])
    expect(store.sessionHistoryError).toBe('时间线暂不可用')
  })

  test('commits historical selection only after timeline and speakers both load', async () => {
    let resolveTimeline: (value: typeof newestTimeline) => void = () => {}
    let resolveClusters: (value: typeof newestClusters) => void = () => {}
    const timelineRequest = new Promise<typeof newestTimeline>(resolve => {
      resolveTimeline = resolve
    })
    const clustersRequest = new Promise<typeof newestClusters>(resolve => {
      resolveClusters = resolve
    })
    const store = useSessionStore()
    store.sessions = [newestSummary, olderSummary]
    store.selectedSessionId = olderSummary.id
    store.timeline = [{ ...newestTimeline[0], id: 'older-span', sessionId: olderSummary.id }]
    const previousTimeline = [...store.timeline]
    vi.spyOn(wordCovenantApi, 'listTimeline').mockReturnValue(timelineRequest)
    vi.spyOn(wordCovenantApi, 'listSpeakerClusters').mockReturnValue(clustersRequest)

    const selection = store.selectSession(newestSummary.id)
    expect(store.selectedSessionId).toBe(olderSummary.id)
    expect(store.timeline).toEqual(previousTimeline)
    expect(store.isSessionHistoryLoading).toBe(true)

    resolveTimeline(newestTimeline)
    await Promise.resolve()
    expect(store.selectedSessionId).toBe(olderSummary.id)

    resolveClusters(newestClusters)
    await expect(selection).resolves.toBe(true)
    expect(store.selectedSessionId).toBe(newestSummary.id)
    expect(store.timeline).toEqual(newestTimeline)
    expect(store.speakerClusters).toEqual(newestClusters)
    expect(store.isSessionHistoryLoading).toBe(false)
  })

  test('preserves the selected archive projection when another session fails to load', async () => {
    const store = useSessionStore()
    store.sessions = [newestSummary, olderSummary]
    store.selectedSessionId = olderSummary.id
    store.timeline = [{ ...newestTimeline[0], id: 'older-span', sessionId: olderSummary.id }]
    const previousTimeline = [...store.timeline]
    vi.spyOn(wordCovenantApi, 'listTimeline').mockRejectedValue(new Error('记录读取失败'))
    vi.spyOn(wordCovenantApi, 'listSpeakerClusters').mockResolvedValue(newestClusters)

    await expect(store.selectSession(newestSummary.id)).resolves.toBe(false)

    expect(store.selectedSessionId).toBe(olderSummary.id)
    expect(store.timeline).toEqual(previousTimeline)
    expect(store.sessionHistoryError).toBe('记录读取失败')
  })

  test('rejects historical selection while recording without loading projections', async () => {
    const store = useSessionStore()
    store.sessions = [newestSummary, olderSummary]
    store.selectedSessionId = newestSummary.id
    store.activeSession = recordingSession
    store.applyCaptureProjection({
      revision: 1,
      status: 'recording',
      permission: 'granted',
      selectedDevice: null,
      devices: [],
      meter: null,
      lastIssue: null,
    })
    const listTimeline = vi.spyOn(wordCovenantApi, 'listTimeline')
    const listSpeakerClusters = vi.spyOn(wordCovenantApi, 'listSpeakerClusters')

    await expect(store.selectSession(olderSummary.id)).resolves.toBe(false)

    expect(store.selectedSessionId).toBe(newestSummary.id)
    expect(listTimeline).not.toHaveBeenCalled()
    expect(listSpeakerClusters).not.toHaveBeenCalled()
  })

  test('deletes the selected session and loads the adjacent archive projection', async () => {
    const store = useSessionStore()
    store.sessions = [newestSummary, olderSummary]
    store.selectedSessionId = newestSummary.id
    store.timeline = newestTimeline
    store.speakerClusters = newestClusters
    const deleteSession = vi.spyOn(wordCovenantApi, 'deleteSession').mockResolvedValue()
    const olderTimeline = [{ ...newestTimeline[0], id: 'older-span', sessionId: olderSummary.id }]
    vi.spyOn(wordCovenantApi, 'listTimeline').mockResolvedValue(olderTimeline)
    vi.spyOn(wordCovenantApi, 'listSpeakerClusters').mockResolvedValue([])

    await expect(store.deleteSession(newestSummary.id)).resolves.toBe(true)

    expect(deleteSession).toHaveBeenCalledWith(newestSummary.id)
    expect(store.sessions).toEqual([olderSummary])
    expect(store.selectedSessionId).toBe(olderSummary.id)
    expect(store.timeline).toEqual(olderTimeline)
    expect(store.speakerClusters).toEqual([])
    expect(store.deletingSessionId).toBeNull()
  })

  test('deletes an unselected session without reloading the current projection', async () => {
    const store = useSessionStore()
    store.sessions = [newestSummary, olderSummary]
    store.selectedSessionId = newestSummary.id
    store.timeline = newestTimeline
    vi.spyOn(wordCovenantApi, 'deleteSession').mockResolvedValue()
    const listTimeline = vi.spyOn(wordCovenantApi, 'listTimeline')

    await expect(store.deleteSession(olderSummary.id)).resolves.toBe(true)

    expect(store.sessions).toEqual([newestSummary])
    expect(store.selectedSessionId).toBe(newestSummary.id)
    expect(store.timeline).toEqual(newestTimeline)
    expect(listTimeline).not.toHaveBeenCalled()
  })

  test('keeps a session when native deletion fails and rejects deletion while recording', async () => {
    const store = useSessionStore()
    store.sessions = [newestSummary]
    store.selectedSessionId = newestSummary.id
    const deleteSession = vi.spyOn(wordCovenantApi, 'deleteSession').mockRejectedValue(new Error('本地数据库忙碌'))

    await expect(store.deleteSession(newestSummary.id)).resolves.toBe(false)
    expect(store.sessions).toEqual([newestSummary])
    expect(store.sessionHistoryError).toBe('本地数据库忙碌')

    store.applyCaptureProjection({
      revision: 1,
      status: 'recording',
      permission: 'granted',
      selectedDevice: null,
      devices: [],
      meter: null,
      lastIssue: null,
    })
    await expect(store.deleteSession(newestSummary.id)).resolves.toBe(false)
    expect(deleteSession).toHaveBeenCalledOnce()
  })

  test('uses the development source and merges incremental synthetic spans by revision', async () => {
    const startDevelopmentMockSession = vi
      .spyOn(wordCovenantApi, 'startDevelopmentMockSession')
      .mockResolvedValue(recordingSession)
    const advanceDevelopmentMock = vi
      .spyOn(wordCovenantApi, 'advanceDevelopmentMock')
      .mockResolvedValueOnce({
        sessionId: recordingSession.id,
        packetsAdvanced: 10,
        exhausted: false,
        spans: [
          {
            id: 'second',
            sessionId: recordingSession.id,
            captureStartNs: 5_000,
            captureEndNs: 6_000,
            speakerClusterId: 'speaker-2',
            text: '第二条模拟转写',
            isFinal: true,
            revision: 1,
            source: 'synthetic' as const,
          },
          {
            id: 'first',
            sessionId: recordingSession.id,
            captureStartNs: 2_000,
            captureEndNs: 4_000,
            speakerClusterId: 'speaker-1',
            text: '第一条模拟转写',
            isFinal: true,
            revision: 1,
            source: 'synthetic' as const,
          },
        ],
      })
      .mockResolvedValueOnce({
        sessionId: recordingSession.id,
        packetsAdvanced: 10,
        exhausted: false,
        spans: [
          {
            id: 'first',
            sessionId: recordingSession.id,
            captureStartNs: 2_000,
            captureEndNs: 4_000,
            speakerClusterId: 'speaker-1',
            text: '第一条模拟转写（修订）',
            isFinal: true,
            revision: 2,
            source: 'synthetic' as const,
          },
        ],
      })
    const store = useSessionStore()

    store.setCaptureInput('development_mock')
    await store.toggleRecording()
    await store.advanceDevelopmentMock()
    await store.advanceDevelopmentMock()

    expect(startDevelopmentMockSession).toHaveBeenCalledOnce()
    expect(advanceDevelopmentMock).toHaveBeenCalledTimes(2)
    expect(store.isDevelopmentMockActive).toBe(true)
    expect(store.timeline.map(span => span.id)).toEqual(['first', 'second'])
    expect(store.timeline[0]?.text).toBe('第一条模拟转写（修订）')
  })

  test('uses the regular stop path when the development script is exhausted', async () => {
    const stoppedSession = { ...recordingSession, state: 'stopped' as const, stoppedAt: '2026-08-08T00:00:12.000Z' }
    vi.spyOn(wordCovenantApi, 'startDevelopmentMockSession').mockResolvedValue(recordingSession)
    vi.spyOn(wordCovenantApi, 'advanceDevelopmentMock').mockResolvedValue({
      sessionId: recordingSession.id,
      packetsAdvanced: 10,
      exhausted: true,
      spans: [],
    })
    const stopSession = vi.spyOn(wordCovenantApi, 'stopSession').mockResolvedValue(stoppedSession)
    const store = useSessionStore()

    store.setCaptureInput('development_mock')
    await store.toggleRecording()
    await store.advanceDevelopmentMock()

    expect(stopSession).toHaveBeenCalledOnce()
    expect(store.activeSession).toEqual(stoppedSession)
    expect(store.isDevelopmentMockActive).toBe(false)
  })

  test('keeps the newest capture projection when events arrive out of order', () => {
    const store = useSessionStore()

    store.applyCaptureProjection({
      revision: 4,
      status: 'recording',
      permission: 'granted',
      selectedDevice: { uid: 'coreaudio:built-in', name: 'MacBook 麦克风' },
      devices: [{ uid: 'coreaudio:built-in', name: 'MacBook 麦克风' }],
      meter: { rmsDbfs: -24, peakDbfs: -12, clipping: false, droppedPackets: 0 },
      lastIssue: null,
    })
    store.applyCaptureProjection({
      revision: 3,
      status: 'failed',
      permission: 'denied',
      selectedDevice: null,
      devices: [],
      meter: null,
      lastIssue: { code: 'permission_denied', deviceName: null },
    })

    expect(store.capture.status).toBe('recording')
    expect(store.isRecording).toBe(true)
    expect(store.capture.meter?.peakDbfs).toBe(-12)
  })

  test('uses the native failed projection when microphone start is rejected', async () => {
    const store = useSessionStore()
    const failedProjection = {
      revision: 1,
      status: 'failed' as const,
      permission: 'denied' as const,
      selectedDevice: null,
      devices: [],
      meter: null,
      lastIssue: { code: 'permission_denied' as const, deviceName: null },
    }
    vi.spyOn(wordCovenantApi, 'startSession').mockRejectedValue(new Error('permission denied'))
    vi.spyOn(wordCovenantApi, 'getCaptureProjection').mockResolvedValue(failedProjection)

    await expect(store.toggleRecording()).resolves.toBeUndefined()

    expect(store.capture).toEqual(failedProjection)
    expect(store.isAwaitingPermission).toBe(false)
    expect(store.isLoading).toBe(false)
  })

  test('falls back to a recoverable failed state when start and projection refresh fail', async () => {
    const store = useSessionStore()
    store.applyCaptureProjection({
      revision: 6,
      status: 'idle',
      permission: 'granted',
      selectedDevice: { uid: 'coreaudio:built-in', name: 'MacBook 麦克风' },
      devices: [{ uid: 'coreaudio:built-in', name: 'MacBook 麦克风' }],
      meter: null,
      lastIssue: null,
    })
    vi.spyOn(wordCovenantApi, 'startSession').mockRejectedValue(new Error('stream start failed'))
    vi.spyOn(wordCovenantApi, 'getCaptureProjection').mockRejectedValue(new Error('projection unavailable'))

    await expect(store.toggleRecording()).resolves.toBeUndefined()

    expect(store.capture).toMatchObject({
      revision: 7,
      status: 'failed',
      meter: null,
      lastIssue: {
        code: 'stream_start_failed',
        deviceName: 'MacBook 麦克风',
      },
    })
    expect(store.isAwaitingPermission).toBe(false)
    expect(store.isLoading).toBe(false)
  })

  test('does not leave an errored start in the awaiting-permission state', async () => {
    const store = useSessionStore()
    vi.spyOn(wordCovenantApi, 'startSession').mockRejectedValue(new Error('permission resolution failed'))
    vi.spyOn(wordCovenantApi, 'getCaptureProjection').mockResolvedValue({
      revision: 1,
      status: 'awaiting_permission',
      permission: 'not_determined',
      selectedDevice: null,
      devices: [],
      meter: null,
      lastIssue: null,
    })

    await expect(store.toggleRecording()).resolves.toBeUndefined()

    expect(store.capture.status).toBe('failed')
    expect(store.capture.lastIssue?.code).toBe('stream_start_failed')
    expect(store.isAwaitingPermission).toBe(false)
  })

  test('turns an idle rejected start into a safe failure and refreshes local ASR availability', async () => {
    const store = useSessionStore()
    const modelStore = useModelStore()
    const rawNativeError = 'failed to open /Users/example/WordCovenant/ggml-base.bin'
    const advancedModel = {
      id: 'advanced-local-model',
      modelKind: 'speech_recognition' as const,
      fileSizeBytes: 2_048_000,
      sha256: 'a'.repeat(64),
      version: 'advanced-local-v1',
      inputFormat: 'whisper.cpp-ggml',
      modelCardId: 'word-covenant/advanced',
      licenseId: 'test-license',
      licenseConfirmedAt: '2026-08-08T00:00:00.000Z',
      importedAt: '2026-08-08T00:00:00.000Z',
    }
    vi.spyOn(wordCovenantApi, 'startSession').mockRejectedValue(new Error(rawNativeError))
    vi.spyOn(wordCovenantApi, 'getCaptureProjection').mockResolvedValue({
      revision: 3,
      status: 'idle',
      permission: 'granted',
      selectedDevice: { uid: 'coreaudio:built-in', name: 'MacBook 麦克风' },
      devices: [{ uid: 'coreaudio:built-in', name: 'MacBook 麦克风' }],
      meter: null,
      lastIssue: null,
    })
    const listLocalModels = vi.spyOn(wordCovenantApi, 'listLocalModels').mockResolvedValue([advancedModel])
    const getActiveLocalAsrProfile = vi
      .spyOn(wordCovenantApi, 'getActiveLocalAsrProfile')
      .mockResolvedValue({ modelId: advancedModel.id })
    const getBundledAsrStatus = vi.spyOn(wordCovenantApi, 'getBundledAsrStatus').mockResolvedValue({
      available: false,
      modelId: 'bundled-model-id',
      message: rawNativeError,
    })

    await store.toggleRecording()

    expect(store.capture.status).toBe('failed')
    expect(store.capture.lastIssue).toEqual({
      code: 'stream_start_failed',
      deviceName: 'MacBook 麦克风',
    })
    expect(JSON.stringify(store.$state)).not.toContain(rawNativeError)
    expect(listLocalModels).toHaveBeenCalledOnce()
    expect(getActiveLocalAsrProfile).toHaveBeenCalledOnce()
    expect(getBundledAsrStatus).toHaveBeenCalledOnce()
    expect(modelStore.models).toEqual([advancedModel])
    expect(modelStore.activeAsrProfile).toEqual({ modelId: advancedModel.id })
    expect(modelStore.bundledAsrStatus).toEqual({
      available: false,
      modelId: null,
      message: '内置离线转写模型不可用，请重新安装应用',
    })
  })

  test('reloads the active timeline and speaker catalog for each newer native final transcript projection', async () => {
    const store = useSessionStore()
    const refreshedTimeline = [
      {
        id: 'native-span-1',
        sessionId: recordingSession.id,
        captureStartNs: 1_000,
        captureEndNs: 2_000,
        speakerClusterId: null,
        text: '已持久化的本地转写',
        isFinal: true,
        revision: 1,
        source: 'local_inference' as const,
      },
    ]
    store.activeSession = recordingSession
    store.selectedSessionId = recordingSession.id
    const listTimeline = vi.spyOn(wordCovenantApi, 'listTimeline').mockResolvedValue(refreshedTimeline)
    const refreshedClusters = [
      {
        ...newestClusters[0]!,
        sessionId: recordingSession.id,
        label: '已登记说话人',
        isUserNamed: true,
      },
    ]
    const listSpeakerClusters = vi.spyOn(wordCovenantApi, 'listSpeakerClusters').mockResolvedValue(refreshedClusters)

    await store.applyFinalTranscriptProjection({ sessionId: recordingSession.id, revision: 2 })
    await store.applyFinalTranscriptProjection({ sessionId: recordingSession.id, revision: 2 })
    await store.applyFinalTranscriptProjection({ sessionId: recordingSession.id, revision: 1 })
    await store.applyFinalTranscriptProjection({ sessionId: 'other-session', revision: 3 })

    expect(listTimeline).toHaveBeenCalledTimes(1)
    expect(listTimeline).toHaveBeenCalledWith(recordingSession.id)
    expect(listSpeakerClusters).toHaveBeenCalledTimes(1)
    expect(listSpeakerClusters).toHaveBeenCalledWith(recordingSession.id)
    expect(store.timeline).toEqual(refreshedTimeline)
    expect(store.speakerClusters).toEqual(refreshedClusters)
    expect(store.finalTranscriptProjectionRevisions).toEqual({
      [recordingSession.id]: 2,
      'other-session': 3,
    })
  })

  test('does not let a slower older final transcript refresh overwrite a newer one', async () => {
    const store = useSessionStore()
    const olderTimeline = [
      {
        id: 'native-span-1',
        sessionId: recordingSession.id,
        captureStartNs: 1_000,
        captureEndNs: 2_000,
        speakerClusterId: null,
        text: '较早快照',
        isFinal: true,
        revision: 1,
        source: 'local_inference' as const,
      },
    ]
    const newerTimeline = [{ ...olderTimeline[0], text: '较新快照', revision: 2 }]
    const olderClusters = [{ ...newestClusters[0]!, sessionId: recordingSession.id, label: '旧名称' }]
    const newerClusters = [{ ...newestClusters[0]!, sessionId: recordingSession.id, label: '自动归类名称' }]
    let resolveOlderTimeline: (timeline: typeof olderTimeline) => void = () => {}
    const olderTimelineRequest = new Promise<typeof olderTimeline>(resolve => {
      resolveOlderTimeline = resolve
    })
    store.activeSession = recordingSession
    store.selectedSessionId = recordingSession.id
    vi.spyOn(wordCovenantApi, 'listTimeline')
      .mockReturnValueOnce(olderTimelineRequest)
      .mockResolvedValueOnce(newerTimeline)
    vi.spyOn(wordCovenantApi, 'listSpeakerClusters')
      .mockResolvedValueOnce(olderClusters)
      .mockResolvedValueOnce(newerClusters)

    const olderRefresh = store.applyFinalTranscriptProjection({
      sessionId: recordingSession.id,
      revision: 4,
    })
    const newerRefresh = store.applyFinalTranscriptProjection({
      sessionId: recordingSession.id,
      revision: 5,
    })
    await newerRefresh
    resolveOlderTimeline(olderTimeline)
    await olderRefresh

    expect(store.timeline).toEqual(newerTimeline)
    expect(store.speakerClusters).toEqual(newerClusters)
  })

  test('forces a final timeline refresh after stopping real microphone capture', async () => {
    const store = useSessionStore()
    const stoppedSession = {
      ...recordingSession,
      state: 'stopped' as const,
      stoppedAt: '2026-08-08T00:00:12.000Z',
    }
    const finalTimeline = [
      {
        id: 'native-final-span',
        sessionId: recordingSession.id,
        captureStartNs: 1_000,
        captureEndNs: 2_000,
        speakerClusterId: null,
        text: '停止前的最终结果',
        isFinal: true,
        revision: 1,
        source: 'local_inference' as const,
      },
    ]
    store.activeSession = recordingSession
    store.selectedSessionId = recordingSession.id
    store.applyCaptureProjection({
      revision: 1,
      status: 'recording',
      permission: 'granted',
      selectedDevice: { uid: 'coreaudio:built-in', name: 'MacBook 麦克风' },
      devices: [{ uid: 'coreaudio:built-in', name: 'MacBook 麦克风' }],
      meter: null,
      lastIssue: null,
    })
    const stopSession = vi.spyOn(wordCovenantApi, 'stopSession').mockResolvedValue(stoppedSession)
    const listTimeline = vi.spyOn(wordCovenantApi, 'listTimeline').mockResolvedValue(finalTimeline)

    await store.toggleRecording()

    expect(stopSession).toHaveBeenCalledOnce()
    expect(listTimeline).toHaveBeenCalledWith(recordingSession.id)
    expect(store.activeSession).toEqual(stoppedSession)
    expect(store.timeline).toEqual(finalTimeline)
  })

  test('waits for a durable speaker result, then reloads the revised timeline', async () => {
    const store = useSessionStore()
    const original = {
      id: 'span-1',
      sessionId: 'session-one',
      captureStartNs: 0,
      captureEndNs: 1_000,
      speakerClusterId: 'speaker-1',
      text: '待校对记录',
      isFinal: true,
      revision: 1,
      source: 'local_inference' as const,
    }
    const clusters = [
      {
        id: 'speaker-2',
        sessionId: 'session-one',
        label: '主持人',
        isUserNamed: true,
        labelRevision: 2,
        aliasRevision: 0,
        mergedIntoClusterId: null,
        canonicalClusterId: 'speaker-2',
        spanCount: 1,
        canEnrollVoiceProfile: true,
      },
    ]
    let completeOperation:
      | ((value: { clusters: typeof clusters; updatedSpans: Array<{ id: string; revision: number }> }) => void)
      | undefined
    const operationResult = new Promise<{
      clusters: typeof clusters
      updatedSpans: Array<{ id: string; revision: number }>
    }>(resolve => {
      completeOperation = resolve
    })
    const revised = {
      ...original,
      speakerClusterId: 'speaker-2',
      revision: 2,
    }
    store.timeline = [original]
    store.selectedSessionId = 'session-one'
    store.speakerClusters = [
      {
        ...clusters[0],
        id: 'speaker-1',
        label: '说话人 1',
        labelRevision: 1,
      },
    ]
    vi.spyOn(wordCovenantApi, 'reassignTranscriptSpeaker').mockReturnValue(operationResult)
    const listTimeline = vi.spyOn(wordCovenantApi, 'listTimeline').mockResolvedValue([revised])

    const pending = store.reassignTranscriptSpeaker({
      sessionId: 'session-one',
      logicalSpanId: original.id,
      expectedRevision: original.revision,
      targetClusterId: 'speaker-2',
    })

    expect(store.isSpeakerOperationPending).toBe(true)
    expect(store.timeline[0]).toEqual(original)
    completeOperation?.({
      clusters,
      updatedSpans: [{ id: original.id, revision: revised.revision }],
    })
    await pending

    expect(listTimeline).toHaveBeenCalledWith('session-one')
    expect(store.speakerClusters).toEqual(clusters)
    expect(store.timeline).toEqual([revised])
    expect(store.isSpeakerOperationPending).toBe(false)
  })

  test('refreshes speaker projections before exposing a stale speaker correction error', async () => {
    const store = useSessionStore()
    const staleSpan = {
      id: 'span-1',
      sessionId: 'session-one',
      captureStartNs: 0,
      captureEndNs: 1_000,
      speakerClusterId: 'speaker-1',
      text: '待校对记录',
      isFinal: true,
      revision: 1,
      source: 'local_inference' as const,
    }
    const refreshedTimeline = [
      {
        ...staleSpan,
        speakerClusterId: 'speaker-2',
        revision: 2,
      },
    ]
    const staleClusters = [
      {
        id: 'speaker-1',
        sessionId: 'session-one',
        label: '说话人 1',
        isUserNamed: false,
        labelRevision: 1,
        aliasRevision: 0,
        mergedIntoClusterId: null,
        canonicalClusterId: 'speaker-1',
        spanCount: 1,
        canEnrollVoiceProfile: true,
      },
    ]
    const refreshedClusters = [
      {
        id: 'speaker-2',
        sessionId: 'session-one',
        label: '主持人',
        isUserNamed: true,
        labelRevision: 2,
        aliasRevision: 0,
        mergedIntoClusterId: null,
        canonicalClusterId: 'speaker-2',
        spanCount: 1,
        canEnrollVoiceProfile: true,
      },
    ]
    let resolveTimeline: (timeline: typeof refreshedTimeline) => void = () => {}
    let resolveClusters: (clusters: typeof refreshedClusters) => void = () => {}
    const timelineRefresh = new Promise<typeof refreshedTimeline>(resolve => {
      resolveTimeline = resolve
    })
    const clusterRefresh = new Promise<typeof refreshedClusters>(resolve => {
      resolveClusters = resolve
    })

    store.timeline = [staleSpan]
    store.selectedSessionId = 'session-one'
    store.speakerClusters = staleClusters
    vi.spyOn(wordCovenantApi, 'reassignTranscriptSpeaker').mockRejectedValue(new Error('记录版本已过期'))
    const listTimeline = vi.spyOn(wordCovenantApi, 'listTimeline').mockReturnValue(timelineRefresh)
    const listSpeakerClusters = vi.spyOn(wordCovenantApi, 'listSpeakerClusters').mockReturnValue(clusterRefresh)

    const pending = store.reassignTranscriptSpeaker({
      sessionId: 'session-one',
      logicalSpanId: staleSpan.id,
      expectedRevision: staleSpan.revision,
      targetClusterId: 'speaker-2',
    })

    await vi.waitFor(() => {
      expect(listTimeline).toHaveBeenCalledWith('session-one')
      expect(listSpeakerClusters).toHaveBeenCalledWith('session-one')
    })
    expect(store.speakerError).toBeNull()
    expect(store.isSpeakerOperationPending).toBe(true)

    resolveTimeline(refreshedTimeline)
    resolveClusters(refreshedClusters)
    await expect(pending).resolves.toBeNull()

    expect(store.timeline).toEqual(refreshedTimeline)
    expect(store.speakerClusters).toEqual(refreshedClusters)
    expect(store.speakerError).toBe('记录版本已过期')
    expect(store.isSpeakerOperationPending).toBe(false)
  })

  test('retains the rejected operation error when its speaker projection refresh fails', async () => {
    const store = useSessionStore()
    const staleSpan = {
      id: 'span-1',
      sessionId: 'session-one',
      captureStartNs: 0,
      captureEndNs: 1_000,
      speakerClusterId: 'speaker-1',
      text: '待校对记录',
      isFinal: true,
      revision: 1,
      source: 'local_inference' as const,
    }
    const staleClusters = [
      {
        id: 'speaker-1',
        sessionId: 'session-one',
        label: '说话人 1',
        isUserNamed: false,
        labelRevision: 1,
        aliasRevision: 0,
        mergedIntoClusterId: null,
        canonicalClusterId: 'speaker-1',
        spanCount: 1,
        canEnrollVoiceProfile: true,
      },
    ]

    store.timeline = [staleSpan]
    store.speakerClusters = staleClusters
    vi.spyOn(wordCovenantApi, 'reassignTranscriptSpeaker').mockRejectedValue(new Error('记录版本已过期'))
    const listTimeline = vi.spyOn(wordCovenantApi, 'listTimeline').mockRejectedValue(new Error('时间线刷新失败'))
    const listSpeakerClusters = vi
      .spyOn(wordCovenantApi, 'listSpeakerClusters')
      .mockRejectedValue(new Error('说话人目录刷新失败'))

    await expect(
      store.reassignTranscriptSpeaker({
        sessionId: 'session-one',
        logicalSpanId: staleSpan.id,
        expectedRevision: staleSpan.revision,
        targetClusterId: 'speaker-2',
      })
    ).resolves.toBeNull()

    expect(listTimeline).toHaveBeenCalledWith('session-one')
    expect(listSpeakerClusters).toHaveBeenCalledWith('session-one')
    expect(store.timeline).toEqual([staleSpan])
    expect(store.speakerClusters).toEqual(staleClusters)
    expect(store.speakerError).toBe('记录版本已过期')
    expect(store.isSpeakerOperationPending).toBe(false)
  })

  test('surfaces plain Tauri speaker errors and translates missing enrollment evidence', async () => {
    const store = useSessionStore()
    store.selectedSessionId = 'session-one'
    vi.spyOn(wordCovenantApi, 'renameSpeakerCluster').mockRejectedValue(
      'this speaker has no high-quality local sample available for enrollment'
    )
    vi.spyOn(wordCovenantApi, 'listTimeline').mockResolvedValue([])
    vi.spyOn(wordCovenantApi, 'listSpeakerClusters').mockResolvedValue([])

    await store.renameSpeakerCluster({
      sessionId: 'session-one',
      clusterId: 'speaker-1',
      expectedLabelRevision: 1,
      label: '主持人',
      consent: true,
    })

    expect(store.speakerError).toBe('该归类还没有可用于声纹学习的录音，请重新录制一段清晰人声后再命名')
  })

  test('keeps an unrecognized plain Tauri speaker error instead of hiding it', async () => {
    const store = useSessionStore()
    store.selectedSessionId = 'session-one'
    vi.spyOn(wordCovenantApi, 'createSpeakerCluster').mockRejectedValue('本地说话人数据库暂时不可用')
    vi.spyOn(wordCovenantApi, 'listTimeline').mockResolvedValue([])
    vi.spyOn(wordCovenantApi, 'listSpeakerClusters').mockResolvedValue([])

    await store.createSpeakerCluster('session-one')

    expect(store.speakerError).toBe('本地说话人数据库暂时不可用')
  })
})
