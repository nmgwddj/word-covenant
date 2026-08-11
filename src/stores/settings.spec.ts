import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, test, vi } from 'vitest'
import { wordCovenantApi } from '@/lib/wordCovenantApi'
import { defaultSpeechDetectionSettings, useSettingsStore } from './settings'

describe('settings store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.restoreAllMocks()
  })

  test('starts with the product default and loads the persisted local threshold', async () => {
    const getSpeechDetectionSettings = vi
      .spyOn(wordCovenantApi, 'getSpeechDetectionSettings')
      .mockResolvedValue({ mode: 'manual', rmsThresholdDbfs: -23 })
    const store = useSettingsStore()

    expect(store.speechDetection).toEqual(defaultSpeechDetectionSettings)

    await store.initialize()

    expect(getSpeechDetectionSettings).toHaveBeenCalledOnce()
    expect(store.speechDetection).toEqual({ mode: 'manual', rmsThresholdDbfs: -23 })
    expect(store.isLoadingSpeechDetection).toBe(false)
    expect(store.speechDetectionError).toBeNull()
  })

  test('retains the -10 dBFS default and presents a safe error when loading fails', async () => {
    vi.spyOn(wordCovenantApi, 'getSpeechDetectionSettings').mockRejectedValue(new Error('native failure'))
    const store = useSettingsStore()

    await store.initialize()

    expect(store.speechDetection).toEqual({ mode: 'adaptive', rmsThresholdDbfs: -10 })
    expect(store.speechDetectionError).toBe('无法读取语音检测设置，已使用默认门限')
    expect(JSON.stringify(store.$state)).not.toContain('native failure')
  })

  test('saves a valid threshold through the typed local API', async () => {
    const setSpeechDetectionSettings = vi
      .spyOn(wordCovenantApi, 'setSpeechDetectionSettings')
      .mockResolvedValue({ mode: 'manual', rmsThresholdDbfs: -18 })
    const store = useSettingsStore()

    await expect(store.setRmsThresholdDbfs(-18)).resolves.toEqual({ mode: 'manual', rmsThresholdDbfs: -18 })

    expect(setSpeechDetectionSettings).toHaveBeenCalledWith({ mode: 'manual', rmsThresholdDbfs: -18 })
    expect(store.speechDetection).toEqual({ mode: 'manual', rmsThresholdDbfs: -18 })
    expect(store.isSavingSpeechDetection).toBe(false)
  })

  test('rejects invalid values before calling the native command', async () => {
    const setSpeechDetectionSettings = vi.spyOn(wordCovenantApi, 'setSpeechDetectionSettings')
    const store = useSettingsStore()

    await expect(store.setRmsThresholdDbfs(-42.5)).resolves.toBeNull()
    await expect(store.setRmsThresholdDbfs(-61)).resolves.toBeNull()
    await expect(store.setRmsThresholdDbfs(1)).resolves.toBeNull()

    expect(setSpeechDetectionSettings).not.toHaveBeenCalled()
    expect(store.speechDetectionError).toBe('门限必须是 -42 到 0 之间的整数 dBFS')
  })

  test('keeps the last confirmed setting when saving fails', async () => {
    vi.spyOn(wordCovenantApi, 'setSpeechDetectionSettings').mockRejectedValue(new Error('native failure'))
    const store = useSettingsStore()
    store.speechDetection = { mode: 'manual', rmsThresholdDbfs: -20 }

    await expect(store.setRmsThresholdDbfs(-32)).resolves.toBeNull()

    expect(store.speechDetection).toEqual({ mode: 'manual', rmsThresholdDbfs: -20 })
    expect(store.speechDetectionError).toBe('无法保存语音检测门限')
    expect(store.isSavingSpeechDetection).toBe(false)
  })
})
