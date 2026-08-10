import { flushPromises, mount } from '@vue/test-utils'
import { describe, expect, test, vi } from 'vitest'
import ModelRegistryPanel from './ModelRegistryPanel.vue'

const model = {
  id: 'model-001',
  modelKind: 'speech_recognition' as const,
  fileSizeBytes: 2_048_000,
  sha256: 'a'.repeat(64),
  version: 'fixture-v1',
  inputFormat: 'whisper.cpp-ggml',
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
    const sourceState = wrapper.get('[data-testid="model-source-path"]')
    expect(sourceState.attributes('data-selected')).toBe('true')
    expect(sourceState.text()).toBe('已选择本地文件')
    expect(wrapper.text()).not.toContain('/local/source/model.gguf')
    await wrapper.get('[data-testid="model-version"]').setValue('fixture-v1')
    await wrapper.get('[data-testid="model-sha256"]').setValue('a'.repeat(64))
    expect((wrapper.get('[data-testid="model-input-format"]').element as HTMLInputElement).value)
      .toBe('whisper.cpp-ggml')

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
        inputFormat: 'whisper.cpp-ggml',
        expectedSha256: 'a'.repeat(64),
        modelCardId: 'word-covenant/fixture',
        licenseId: 'test-license',
        licenseAcknowledged: true,
      },
    ]])
  })

  test('keeps a selected local file when the native picker is cancelled and clears errors when dismissed', async () => {
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
    expect(wrapper.get('[data-testid="model-source-path"]').text()).toBe('已选择本地文件')
    expect(wrapper.text()).not.toContain('/local/source/model.gguf')

    await wrapper.get('[data-testid="close-model-import"]').trigger('click')
    expect(wrapper.find('[data-testid="model-source-path"]').exists()).toBe(false)
    await wrapper.get('[data-testid="open-model-import"]').trigger('click')
    expect(wrapper.find('[role="alert"]').exists()).toBe(false)
    expect(wrapper.get('[data-testid="model-source-path"]').text()).toBe('尚未选择文件')
  })

  test('shows the active local ASR profile and only exposes whisper.cpp-ggml ASR models for selection', async () => {
    const compatibleAlternative = {
      ...model,
      id: 'model-002',
      sha256: 'b'.repeat(64),
      version: 'fixture-v2',
    }
    const incompatibleFormat = {
      ...model,
      id: 'model-003',
      sha256: 'c'.repeat(64),
      version: 'not-for-whisper',
      inputFormat: 'gguf',
    }
    const nonAsr = {
      ...model,
      id: 'model-004',
      sha256: 'd'.repeat(64),
      version: 'vad-fixture',
      modelKind: 'voice_activity_detection' as const,
    }
    const wrapper = mount(ModelRegistryPanel, {
      props: {
        models: [model, compatibleAlternative, incompatibleFormat, nonAsr],
        compatibleAsrModels: [model, compatibleAlternative],
        activeAsrProfile: { modelId: model.id },
      },
    })

    expect(wrapper.get('[data-testid="active-asr-profile"]').text()).toContain('fixture-v1')
    expect(wrapper.get('[data-testid="active-asr-profile"]').text()).toContain('whisper.cpp-ggml')
    const selector = wrapper.get('[data-testid="active-asr-model-select"]')
    expect(selector.findAll('option').map(option => option.text())).toEqual([
      '选择兼容模型',
      `fixture-v1 · ${'a'.repeat(8)}`,
      `fixture-v2 · ${'b'.repeat(8)}`,
    ])

    await selector.setValue(compatibleAlternative.id)

    expect(wrapper.emitted('selectActiveAsrModel')).toEqual([[compatibleAlternative.id]])
    expect(wrapper.text()).toContain('当前转写')
    expect(wrapper.text()).not.toContain('/local/')
  })
})
