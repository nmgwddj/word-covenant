import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, test, vi } from 'vitest'
import { wordCovenantApi } from '@/lib/wordCovenantApi'
import { useModelStore } from './models'

const importedModel = {
  id: 'model-001',
  modelKind: 'speech_recognition' as const,
  fileSizeBytes: 2_048_000,
  sha256: 'a'.repeat(64),
  version: 'fixture-v1',
  inputFormat: 'gguf',
  modelCardId: 'word-covenant/fixture',
  licenseId: 'test-license',
  licenseConfirmedAt: '2026-08-08T00:00:00.000Z',
  importedAt: '2026-08-08T00:00:00.000Z',
}

describe('model store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.restoreAllMocks()
  })

  test('loads only local model registrations during initialization', async () => {
    vi.spyOn(wordCovenantApi, 'listLocalModels').mockResolvedValue([importedModel])
    const store = useModelStore()

    await store.initialize()

    expect(store.models).toEqual([importedModel])
    expect(store.isLoading).toBe(false)
  })

  test('records a successful local import without changing egress state', async () => {
    const importLocalModel = vi.spyOn(wordCovenantApi, 'importLocalModel').mockResolvedValue(importedModel)
    const store = useModelStore()

    await store.importLocalModel({
      sourcePath: '/local/source/model.gguf',
      modelKind: 'speech_recognition',
      version: 'fixture-v1',
      inputFormat: 'gguf',
      expectedSha256: 'a'.repeat(64),
      modelCardId: 'word-covenant/fixture',
      licenseId: 'test-license',
      licenseAcknowledged: true,
    })

    expect(importLocalModel).toHaveBeenCalledOnce()
    expect(store.models).toEqual([importedModel])
    expect(store.importError).toBeNull()
    expect(store.isImporting).toBe(false)
  })

  test('clears stale import errors before native local model selection, including cancellation', async () => {
    const selectLocalModelFile = vi.spyOn(wordCovenantApi, 'selectLocalModelFile').mockResolvedValue(null)
    const store = useModelStore()
    store.importError = 'SHA-256 不匹配'

    await expect(store.selectLocalModelFile()).resolves.toBeNull()

    expect(selectLocalModelFile).toHaveBeenCalledOnce()
    expect(store.importError).toBeNull()
  })

  test('clears the visible import error when the form is dismissed', () => {
    const store = useModelStore()
    store.importError = 'SHA-256 不匹配'

    store.clearImportError()

    expect(store.importError).toBeNull()
  })

  test('keeps an import failure visible for correction', async () => {
    vi.spyOn(wordCovenantApi, 'importLocalModel').mockRejectedValue(new Error('SHA-256 不匹配'))
    const store = useModelStore()

    await expect(store.importLocalModel({
      sourcePath: '/local/source/model.gguf',
      modelKind: 'speech_recognition',
      version: 'fixture-v1',
      inputFormat: 'gguf',
      expectedSha256: 'b'.repeat(64),
      modelCardId: 'word-covenant/fixture',
      licenseId: 'test-license',
      licenseAcknowledged: true,
    })).rejects.toThrow('SHA-256 不匹配')

    expect(store.models).toEqual([])
    expect(store.importError).toBe('SHA-256 不匹配')
    expect(store.isImporting).toBe(false)
  })
})
