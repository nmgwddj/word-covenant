import { mount } from '@vue/test-utils'
import { describe, expect, test } from 'vitest'
import CaptureSettingsPanel from './CaptureSettingsPanel.vue'

const settings = { rmsThresholdDbfs: -10 }
const meter = {
  rmsDbfs: -21.6,
  peakDbfs: -5.2,
  clipping: false,
  droppedPackets: 0,
}

function mountPanel(overrides: Record<string, unknown> = {}) {
  return mount(CaptureSettingsPanel, {
    attachTo: document.body,
    props: {
      settings,
      meter,
      ...overrides,
    },
  })
}

describe('CaptureSettingsPanel', () => {
  test('shows the default threshold together with RMS and peak tuning values', () => {
    const wrapper = mountPanel()

    expect(wrapper.get('[data-testid="speech-threshold-value"]').text()).toBe('-10 dBFS')
    expect(wrapper.get('[data-testid="speech-threshold-range"]').attributes('min')).toBe('-60')
    expect(wrapper.get('[data-testid="speech-threshold-range"]').attributes('max')).toBe('0')
    expect(wrapper.get('[data-testid="speech-threshold-number"]').attributes('step')).toBe('1')
    expect(wrapper.get('[data-testid="capture-settings-rms"]').text()).toBe('-22 dBFS')
    expect(wrapper.get('[data-testid="capture-settings-peak"]').text()).toBe('-5 dBFS')
    wrapper.unmount()
  })

  test('saves the completed slider adjustment and clears a prior error', async () => {
    const wrapper = mountPanel({ error: '无法保存语音检测门限' })

    await wrapper.get('[data-testid="speech-threshold-range"]').setValue('-28')

    expect(wrapper.emitted('clearError')).toEqual([[]])
    expect(wrapper.emitted('save')).toEqual([[-28]])
    wrapper.unmount()
  })

  test('saves a completed numeric threshold adjustment', async () => {
    const wrapper = mountPanel()

    await wrapper.get('[data-testid="speech-threshold-number"]').setValue('-19')

    expect(wrapper.emitted('save')).toEqual([[-19]])
    wrapper.unmount()
  })

  test('resets the threshold to the -10 dBFS product default', async () => {
    const wrapper = mountPanel({ settings: { rmsThresholdDbfs: -34 } })

    await wrapper.get('[data-testid="reset-speech-threshold"]').trigger('click')

    expect(wrapper.get('[data-testid="speech-threshold-value"]').text()).toBe('-10 dBFS')
    expect(wrapper.emitted('save')).toEqual([[-10]])
    wrapper.unmount()
  })

  test('locks threshold controls while a capture is preparing or recording', () => {
    const wrapper = mountPanel({ locked: true })

    expect(wrapper.get('[data-testid="speech-threshold-range"]').attributes('disabled')).toBeDefined()
    expect(wrapper.get('[data-testid="speech-threshold-number"]').attributes('disabled')).toBeDefined()
    expect(wrapper.get('[data-testid="reset-speech-threshold"]').attributes('disabled')).toBeDefined()
    expect(wrapper.get('[data-testid="speech-threshold-lock-message"]').text()).toContain('本次会话')
    wrapper.unmount()
  })

  test('shows loading, saving, and error feedback without audio content', () => {
    const loading = mountPanel({ loading: true, meter: null })
    expect(loading.get('[role="status"]').text()).toBe('正在读取本地设置')
    expect(loading.get('[data-testid="capture-settings-rms"]').text()).toBe('-- dBFS')
    loading.unmount()

    const saving = mountPanel({ saving: true, error: '无法保存语音检测门限' })
    expect(saving.get('[role="status"]').text()).toBe('正在保存本地设置')
    expect(saving.get('[role="alert"]').text()).toBe('无法保存语音检测门限')
    saving.unmount()
  })

  test('closes from the close button or Escape', async () => {
    const wrapper = mountPanel()

    await wrapper.get('[aria-label="关闭录音检测设置"]').trigger('click')
    await wrapper.get('aside').trigger('keydown', { key: 'Escape' })

    expect(wrapper.emitted('close')).toEqual([[], []])
    wrapper.unmount()
  })
})
