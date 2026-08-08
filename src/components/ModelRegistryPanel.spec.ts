import { flushPromises, mount } from '@vue/test-utils'
import { describe, expect, test, vi } from 'vitest'
import ModelRegistryPanel from './ModelRegistryPanel.vue'

const model = {
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

describe('ModelRegistryPanel', () => {
  test('shows only local model provenance and opens an explicit import form', async () => {
    const wrapper = mount(ModelRegistryPanel, {
      props: { models: [model] },
    })

    expect(wrapper.text()).toContain('本地模型')
    expect(wrapper.text()).toContain('语音识别')
    expect(wrapper.text()).toContain('fixture-v1')
    expect(wrapper.find('[data-testid="model-source-path"]').exists()).toBe(false)

    await wrapper.get('[data-testid="open-model-import"]').trigger('click')

    expect(wrapper.find('[data-testid="model-source-path"]').exists()).toBe(true)
    expect(wrapper.get('[data-testid="submit-model-import"]').attributes('disabled')).toBeDefined()
  })

  test('requires a license acknowledgement before emitting a local import request', async () => {
    const selectSourcePath = vi.fn().mockResolvedValue('/local/source/model.gguf')
    const wrapper = mount(ModelRegistryPanel, {
      props: { models: [], selectSourcePath },
    })
    await wrapper.get('[data-testid="open-model-import"]').trigger('click')
    await wrapper.get('[data-testid="choose-model-source"]').trigger('click')
    await flushPromises()
    expect(selectSourcePath).toHaveBeenCalledOnce()
    expect((wrapper.get('[data-testid="model-source-path"]').element as HTMLInputElement).value)
      .toBe('/local/source/model.gguf')
    await wrapper.get('[data-testid="model-version"]').setValue('fixture-v1')
    await wrapper.get('[data-testid="model-sha256"]').setValue('a'.repeat(64))

    await wrapper.get('[data-testid="model-card-id"]').setValue('word-covenant/fixture')
    await wrapper.get('[data-testid="model-license-id"]').setValue('test-license')
    expect(wrapper.get('[data-testid="submit-model-import"]').attributes('disabled')).toBeDefined()

    await wrapper.get('[data-testid="model-license-acknowledged"]').setValue(true)
    await wrapper.get('form').trigger('submit')

    expect(wrapper.emitted('import')).toEqual([[
      {
        sourcePath: '/local/source/model.gguf',
        modelKind: 'speech_recognition',
        version: 'fixture-v1',
        inputFormat: 'gguf',
        expectedSha256: 'a'.repeat(64),
        modelCardId: 'word-covenant/fixture',
        licenseId: 'test-license',
        licenseAcknowledged: true,
      },
    ]])
  })

  test('keeps a selected path when the native picker is cancelled and clears errors when dismissed', async () => {
    const selectSourcePath = vi.fn()
      .mockResolvedValueOnce('/local/source/model.gguf')
      .mockResolvedValueOnce(null)
    const wrapper = mount(ModelRegistryPanel, {
      props: {
        models: [],
        error: 'SHA-256 不匹配',
        selectSourcePath,
      },
    })

    await wrapper.get('[data-testid="open-model-import"]').trigger('click')
    expect(wrapper.emitted('clearError')).toHaveLength(1)
    await wrapper.setProps({ error: null })
    expect(wrapper.find('[role="alert"]').exists()).toBe(false)

    await wrapper.get('[data-testid="choose-model-source"]').trigger('click')
    await flushPromises()
    await wrapper.get('[data-testid="choose-model-source"]').trigger('click')
    await flushPromises()
    expect(selectSourcePath).toHaveBeenCalledTimes(2)
    expect((wrapper.get('[data-testid="model-source-path"]').element as HTMLInputElement).value)
      .toBe('/local/source/model.gguf')

    await wrapper.get('[data-testid="close-model-import"]').trigger('click')
    expect(wrapper.find('[data-testid="model-source-path"]').exists()).toBe(false)
    await wrapper.get('[data-testid="open-model-import"]').trigger('click')
    expect(wrapper.find('[role="alert"]').exists()).toBe(false)
    expect((wrapper.get('[data-testid="model-source-path"]').element as HTMLInputElement).value).toBe('')
  })
})
