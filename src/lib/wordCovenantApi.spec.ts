import { afterEach, describe, expect, test, vi } from 'vitest'
import { invoke } from '@tauri-apps/api/core'

vi.mock('@tauri-apps/api/core', () => ({
  invoke: vi.fn(),
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
    expect((await wordCovenantApi.listTimeline(session.id))).toHaveLength(3)
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  test('keeps microphone projection APIs inert in browser preview', async () => {
    const fetchSpy = vi.fn()
    vi.stubGlobal('fetch', fetchSpy)
    const { wordCovenantApi } = await loadBrowserApi()

    const projection = await wordCovenantApi.getCaptureProjection()
    const unlisten = await wordCovenantApi.onCaptureProjection(vi.fn())

    expect(projection.status).toBe('idle')
    expect(projection.devices).toEqual([])
    expect(unlisten()).toBeUndefined()
    await expect(wordCovenantApi.startSession()).rejects.toThrow('浏览器预览不提供真实麦克风输入')
    await expect(wordCovenantApi.selectInputDevice('coreaudio:demo')).rejects.toThrow('浏览器预览')
    expect((await wordCovenantApi.getPrivacyStatus()).recordingSessionId).toBeNull()
    expect(await wordCovenantApi.listLocalModels()).toEqual([])
    await expect(wordCovenantApi.selectLocalModelFile()).rejects.toThrow(
      '浏览器预览不能打开本机模型文件选择器',
    )
    await expect(wordCovenantApi.importLocalModel({
      sourcePath: '/local/model.gguf',
      modelKind: 'speech_recognition',
      version: 'fixture-v1',
      inputFormat: 'gguf',
      expectedSha256: 'a'.repeat(64),
      modelCardId: 'fixture',
      licenseId: 'test-license',
      licenseAcknowledged: true,
    })).rejects.toThrow('浏览器预览不能导入本地模型文件')
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  test('uses the typed native command for local model selection and preserves cancellation', async () => {
    vi.stubGlobal('__TAURI_INTERNALS__', {})
    const invokeMock = vi.mocked(invoke)
    invokeMock
      .mockResolvedValueOnce('/Users/example/Models/whisper-small.gguf')
      .mockResolvedValueOnce(null)
    const { wordCovenantApi } = await loadBrowserApi()

    await expect(wordCovenantApi.selectLocalModelFile()).resolves.toBe(
      '/Users/example/Models/whisper-small.gguf',
    )
    await expect(wordCovenantApi.selectLocalModelFile()).resolves.toBeNull()
    expect(invokeMock).toHaveBeenNthCalledWith(1, 'select_local_model_file')
    expect(invokeMock).toHaveBeenNthCalledWith(2, 'select_local_model_file')
  })
})
