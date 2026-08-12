import { mount } from '@vue/test-utils'
import { describe, expect, test } from 'vitest'
import SettingsPage from './SettingsPage.vue'

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

const capture = {
  revision: 1,
  status: 'idle' as const,
  permission: 'granted' as const,
  selectedDevice: { uid: 'coreaudio:built-in', name: 'MacBook 麦克风' },
  devices: [
    { uid: 'coreaudio:built-in', name: 'MacBook 麦克风' },
    { uid: 'coreaudio:usb', name: 'USB 麦克风' },
  ],
  meter: null,
  lastIssue: null,
}

const voiceProfile = {
  id: 'profile-one',
  revision: 2,
  displayName: '主持人',
  state: 'learning' as const,
  confirmedDurationNs: 2_000_000_000,
  readyConfirmedDurationNs: 4_000_000_000,
  modelId: 'campplus',
  modelVersion: '2024-10-14',
  lastConfirmationAt: null,
  canAddConfirmedSample: false,
  updatedAt: '2026-08-12T08:00:00.000Z',
}

describe('SettingsPage', () => {
  test('keeps local model maintenance in a dedicated settings view', async () => {
    const wrapper = mount(SettingsPage, {
      props: {
        models: [model],
        capture,
        compatibleAsrModels: [model],
        activeAsrProfile: { modelId: model.id },
        voiceProfiles: [voiceProfile],
      },
    })

    expect(wrapper.get('#settings-page-title').text()).toBe('设置')
    expect(wrapper.get('#recording-settings-title').text()).toBe('录音与检测')
    expect(wrapper.get('#model-settings-title').text()).toBe('模型与转写')
    expect(wrapper.get('#voice-profile-settings-title').text()).toBe('声纹档案')
    expect(wrapper.get('#voice-profile-name-profile-one').attributes('value')).toBe('主持人')
    expect(wrapper.find('[data-testid="input-device-select"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="speech-mode-adaptive"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="active-asr-model-select"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="open-model-import"]').exists()).toBe(true)
    expect(wrapper.text()).not.toContain('当前会话')

    await wrapper.get('[data-testid="input-device-select"]').trigger('click')
    await wrapper.get('[role="option"][data-value="coreaudio:usb"]').trigger('click')
    await wrapper.get('[data-testid="speech-mode-manual"]').trigger('click')
    await wrapper.get('[data-testid="refresh-input-devices"]').trigger('click')
    await wrapper.get('[aria-label="返回工作台"]').trigger('click')

    expect(wrapper.emitted('selectInputDevice')).toEqual([['coreaudio:usb']])
    expect(wrapper.emitted('refreshInputDevices')).toEqual([[]])
    expect(wrapper.emitted('saveSpeechDetection')).toEqual([[{ mode: 'manual', rmsThresholdDbfs: -10 }]])
    expect(wrapper.emitted('close')).toEqual([[]])
  })
})
