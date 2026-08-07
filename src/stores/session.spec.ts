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
})
