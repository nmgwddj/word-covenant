import { afterEach, describe, expect, test, vi } from 'vitest'
import { invoke } from '@tauri-apps/api/core'
import { listen } from '@tauri-apps/api/event'

vi.mock('@tauri-apps/api/core', () => ({
  invoke: vi.fn(),
}))

vi.mock('@tauri-apps/api/event', () => ({
  listen: vi.fn(),
}))

async function loadBrowserApi() {
  vi.resetModules()
  return import('./wordCovenantApi')
}

describe('wordCovenantApi browser development mock', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.clearAllMocks()
  })

  test('advances a local scripted session without requesting microphone or network access', async () => {
    const fetchSpy = vi.fn()
    vi.stubGlobal('fetch', fetchSpy)
    const { wordCovenantApi } = await loadBrowserApi()

    const session = await wordCovenantApi.startDevelopmentMockSession()
    let progress = await wordCovenantApi.advanceDevelopmentMock()
    expect(progress.spans).toEqual([])

    for (let tick = 1; tick < 14; tick += 1) {
      progress = await wordCovenantApi.advanceDevelopmentMock()
    }

    expect(progress.sessionId).toBe(session.id)
    expect(progress.spans).toHaveLength(1)
    expect(progress.spans[0]?.source).toBe('synthetic')

    for (let tick = 14; tick < 60; tick += 1) {
      progress = await wordCovenantApi.advanceDevelopmentMock()
    }

    expect(progress.exhausted).toBe(true)
    expect(await wordCovenantApi.listTimeline(session.id)).toHaveLength(3)
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  test('keeps microphone projection APIs inert in browser preview', async () => {
    const fetchSpy = vi.fn()
    vi.stubGlobal('fetch', fetchSpy)
    const { wordCovenantApi } = await loadBrowserApi()

    const projection = await wordCovenantApi.getCaptureProjection()
    const unlisten = await wordCovenantApi.onCaptureProjection(vi.fn())
    const unlistenFinalTranscript = await wordCovenantApi.onFinalTranscriptProjection(vi.fn())

    expect(projection.status).toBe('idle')
    expect(projection.devices).toEqual([])
    expect(projection.bridge).toBeNull()
    expect(unlisten()).toBeUndefined()
    expect(unlistenFinalTranscript()).toBeUndefined()
    await expect(wordCovenantApi.startSession()).rejects.toThrow('浏览器预览不提供真实麦克风输入')
    await expect(wordCovenantApi.selectInputDevice('coreaudio:demo')).rejects.toThrow('浏览器预览')
    expect((await wordCovenantApi.getPrivacyStatus()).recordingSessionId).toBeNull()
    expect(await wordCovenantApi.listLocalModels()).toEqual([])
    expect(await wordCovenantApi.getBundledAsrStatus()).toEqual({
      available: false,
      modelId: null,
      message: '浏览器预览不包含内置本地转写模型',
    })
    await expect(wordCovenantApi.selectLocalModelFile()).rejects.toThrow('浏览器预览不能打开本机模型文件选择器')
    await expect(
      wordCovenantApi.importLocalModel({
        sourcePath: '/local/model.gguf',
        modelKind: 'speech_recognition',
        version: 'fixture-v1',
        inputFormat: 'gguf',
        expectedSha256: 'a'.repeat(64),
        modelCardId: 'fixture',
        licenseId: 'test-license',
        licenseAcknowledged: true,
      })
    ).rejects.toThrow('浏览器预览不能导入本地模型文件')
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  test('subscribes to the compact native final transcript projection event', async () => {
    vi.stubGlobal('__TAURI_INTERNALS__', {})
    const nativeUnlisten = vi.fn()
    vi.mocked(listen).mockResolvedValue(nativeUnlisten)
    const { wordCovenantApi } = await loadBrowserApi()

    await expect(wordCovenantApi.onFinalTranscriptProjection(vi.fn())).resolves.toBe(nativeUnlisten)

    expect(listen).toHaveBeenCalledWith('final-transcript-projection', expect.any(Function))
  })

  test('uses the typed native command for local model selection and preserves cancellation', async () => {
    vi.stubGlobal('__TAURI_INTERNALS__', {})
    const invokeMock = vi.mocked(invoke)
    invokeMock.mockResolvedValueOnce('/Users/example/Models/whisper-small.gguf').mockResolvedValueOnce(null)
    const { wordCovenantApi } = await loadBrowserApi()

    await expect(wordCovenantApi.selectLocalModelFile()).resolves.toBe('/Users/example/Models/whisper-small.gguf')
    await expect(wordCovenantApi.selectLocalModelFile()).resolves.toBeNull()
    expect(invokeMock).toHaveBeenNthCalledWith(1, 'select_local_model_file')
    expect(invokeMock).toHaveBeenNthCalledWith(2, 'select_local_model_file')
  })

  test('uses the typed native command for bundled local ASR status', async () => {
    vi.stubGlobal('__TAURI_INTERNALS__', {})
    const invokeMock = vi.mocked(invoke)
    const status = {
      available: true,
      modelId: 'b1a7f91b-0799-4af1-97e9-0d55aa8a5b9b',
      message: null,
    }
    invokeMock.mockResolvedValue(status)
    const { wordCovenantApi } = await loadBrowserApi()

    await expect(wordCovenantApi.getBundledAsrStatus()).resolves.toEqual(status)

    expect(invokeMock).toHaveBeenCalledWith('get_bundled_asr_status')
  })

  test('keeps browser speech detection settings in local memory without network access', async () => {
    const fetchSpy = vi.fn()
    vi.stubGlobal('fetch', fetchSpy)
    const { wordCovenantApi } = await loadBrowserApi()

    await expect(wordCovenantApi.getSpeechDetectionSettings()).resolves.toEqual({ mode: 'adaptive', rmsThresholdDbfs: -10 })
    await expect(wordCovenantApi.setSpeechDetectionSettings({ mode: 'manual', rmsThresholdDbfs: -26 })).resolves.toEqual({
      mode: 'manual',
      rmsThresholdDbfs: -26,
    })
    await expect(wordCovenantApi.getSpeechDetectionSettings()).resolves.toEqual({ mode: 'manual', rmsThresholdDbfs: -26 })

    expect(fetchSpy).not.toHaveBeenCalled()
  })

  test('uses typed native commands for speech detection settings', async () => {
    vi.stubGlobal('__TAURI_INTERNALS__', {})
    const invokeMock = vi.mocked(invoke)
    invokeMock
      .mockResolvedValueOnce({ mode: 'adaptive', rmsThresholdDbfs: -10 })
      .mockResolvedValueOnce({ mode: 'manual', rmsThresholdDbfs: -18 })
    const { wordCovenantApi } = await loadBrowserApi()

    await expect(wordCovenantApi.getSpeechDetectionSettings()).resolves.toEqual({ mode: 'adaptive', rmsThresholdDbfs: -10 })
    await expect(wordCovenantApi.setSpeechDetectionSettings({ mode: 'manual', rmsThresholdDbfs: -18 })).resolves.toEqual({
      mode: 'manual',
      rmsThresholdDbfs: -18,
    })

    expect(invokeMock).toHaveBeenNthCalledWith(1, 'get_speech_detection_settings')
    expect(invokeMock).toHaveBeenNthCalledWith(2, 'set_speech_detection_settings', {
      input: { mode: 'manual', rmsThresholdDbfs: -18 },
    })
  })

  test('keeps browser speaker corrections local and returns durable revisions', async () => {
    const fetchSpy = vi.fn()
    vi.stubGlobal('fetch', fetchSpy)
    const { wordCovenantApi } = await loadBrowserApi()

    const initialClusters = await wordCovenantApi.listSpeakerClusters('local-demo-session')
    const speakerTwo = initialClusters.find(cluster => cluster.id === 'speaker-2')
    expect(speakerTwo?.label).toBe('说话人 2')

    const renamed = await wordCovenantApi.renameSpeakerCluster({
      sessionId: 'local-demo-session',
      clusterId: 'speaker-2',
      expectedLabelRevision: speakerTwo!.labelRevision,
      label: '主持人',
    })
    expect(renamed.clusters.find(cluster => cluster.id === 'speaker-2')).toMatchObject({
      label: '主持人',
      isUserNamed: true,
      labelRevision: 2,
    })

    const reassigned = await wordCovenantApi.reassignTranscriptSpeaker({
      sessionId: 'local-demo-session',
      logicalSpanId: 'span-002',
      expectedRevision: 1,
      targetClusterId: 'speaker-1',
    })
    expect(reassigned.updatedSpans).toEqual([{ id: 'span-002', revision: 2 }])
    expect(
      (await wordCovenantApi.listTimeline('local-demo-session')).find(span => span.id === 'span-002')
    ).toMatchObject({ speakerClusterId: 'speaker-1', revision: 2 })

    const created = await wordCovenantApi.createSpeakerCluster({
      sessionId: 'local-demo-session',
    })
    expect(created.clusters.map(cluster => cluster.id)).toContain('speaker-3')
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  test('uses typed native speaker commands', async () => {
    vi.stubGlobal('__TAURI_INTERNALS__', {})
    const invokeMock = vi.mocked(invoke)
    const clusters = [
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
    invokeMock
      .mockResolvedValueOnce(clusters)
      .mockResolvedValueOnce({ clusters, updatedSpans: [] })
      .mockResolvedValueOnce({ clusters, updatedSpans: [] })
      .mockResolvedValueOnce({ clusters, updatedSpans: [{ id: 'span-1', revision: 2 }] })
    const { wordCovenantApi } = await loadBrowserApi()

    await expect(wordCovenantApi.listSpeakerClusters('session-one')).resolves.toEqual(clusters)
    await wordCovenantApi.createSpeakerCluster({ sessionId: 'session-one' })
    await wordCovenantApi.renameSpeakerCluster({
      sessionId: 'session-one',
      clusterId: 'speaker-1',
      expectedLabelRevision: 1,
      label: '主持人',
    })
    await wordCovenantApi.reassignTranscriptSpeaker({
      sessionId: 'session-one',
      logicalSpanId: 'span-1',
      expectedRevision: 1,
      targetClusterId: null,
    })

    expect(invokeMock).toHaveBeenNthCalledWith(1, 'list_speaker_clusters', { sessionId: 'session-one' })
    expect(invokeMock).toHaveBeenNthCalledWith(2, 'create_speaker_cluster', {
      input: { sessionId: 'session-one' },
    })
    expect(invokeMock).toHaveBeenNthCalledWith(3, 'rename_speaker_cluster', {
      input: {
        sessionId: 'session-one',
        clusterId: 'speaker-1',
        expectedLabelRevision: 1,
        label: '主持人',
      },
    })
    expect(invokeMock).toHaveBeenNthCalledWith(4, 'reassign_transcript_speaker', {
      input: {
        sessionId: 'session-one',
        logicalSpanId: 'span-1',
        expectedRevision: 1,
        targetClusterId: null,
      },
    })
  })
})
