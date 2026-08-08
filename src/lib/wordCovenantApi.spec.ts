import { afterEach, describe, expect, test, vi } from 'vitest'

async function loadBrowserApi() {
  vi.resetModules()
  return import('./wordCovenantApi')
}

describe('wordCovenantApi browser development mock', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
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
    expect(fetchSpy).not.toHaveBeenCalled()
  })
})
