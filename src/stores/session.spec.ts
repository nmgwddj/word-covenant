import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, test, vi } from 'vitest'
import { wordCovenantApi } from '@/lib/wordCovenantApi'
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
    const startDevelopmentMockSession = vi.spyOn(wordCovenantApi, 'startDevelopmentMockSession')
      .mockResolvedValue(recordingSession)
    const advanceDevelopmentMock = vi.spyOn(wordCovenantApi, 'advanceDevelopmentMock')
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
    expect(store.timeline.map((span) => span.id)).toEqual(['first', 'second'])
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
})
