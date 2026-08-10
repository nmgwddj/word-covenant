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

describe('session store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.restoreAllMocks()
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

  test('reloads the active timeline once for each newer native final transcript projection', async () => {
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
    const listTimeline = vi.spyOn(wordCovenantApi, 'listTimeline').mockResolvedValue(refreshedTimeline)

    await store.applyFinalTranscriptProjection({ sessionId: recordingSession.id, revision: 2 })
    await store.applyFinalTranscriptProjection({ sessionId: recordingSession.id, revision: 2 })
    await store.applyFinalTranscriptProjection({ sessionId: recordingSession.id, revision: 1 })
    await store.applyFinalTranscriptProjection({ sessionId: 'other-session', revision: 3 })

    expect(listTimeline).toHaveBeenCalledTimes(1)
    expect(listTimeline).toHaveBeenCalledWith(recordingSession.id)
    expect(store.timeline).toEqual(refreshedTimeline)
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
    let resolveOlderTimeline: (timeline: typeof olderTimeline) => void = () => {}
    const olderTimelineRequest = new Promise<typeof olderTimeline>(resolve => {
      resolveOlderTimeline = resolve
    })
    store.activeSession = recordingSession
    vi.spyOn(wordCovenantApi, 'listTimeline')
      .mockReturnValueOnce(olderTimelineRequest)
      .mockResolvedValueOnce(newerTimeline)

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
})
