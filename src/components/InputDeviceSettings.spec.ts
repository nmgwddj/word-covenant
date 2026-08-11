import { mount } from '@vue/test-utils'
import { describe, expect, test } from 'vitest'
import type { CaptureBridgeProjection } from '@/types'
import InputDeviceSettings from './InputDeviceSettings.vue'

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

const bridge: CaptureBridgeProjection = {
  status: 'armed',
  armed: true,
  shutdownRequested: false,
  workerFinished: false,
  metrics: {
    ingressPacketsConsumed: 0,
    ingressDiscontinuities: 0,
    segmenterFailures: 0,
    jobsAdmitted: 0,
    jobsCompleted: 0,
    jobQueueSaturated: 0,
    resultQueueSaturated: 0,
    unavailableEngineOutcomes: 0,
    engineFailureOutcomes: 0,
    shutdownOutcomes: 0,
    outcomeClaimsAborted: 0,
    jobQueueHighWatermark: 0,
    resultQueueHighWatermark: 0,
    pendingEventHighWatermark: 0,
    jobQueueDepth: 0,
    resultQueueDepth: 0,
    pendingEventDepth: 0,
    workerHoldsOutcome: false,
    ownedOutcomeLeaseActive: false,
    closing: false,
  },
}

describe('InputDeviceSettings', () => {
  test('emits a stable input device UID when idle', async () => {
    const wrapper = mount(InputDeviceSettings, { props: { capture } })

    const selector = wrapper.get('[data-testid="input-device-select"]')
    expect(selector.text()).toContain('MacBook 麦克风')
    await selector.trigger('click')
    await wrapper.get('[role="option"][data-value="coreaudio:usb"]').trigger('click')

    expect(wrapper.emitted('select')).toEqual([['coreaudio:usb']])
  })

  test('keeps the refresh action available when no input devices are detected', async () => {
    const wrapper = mount(InputDeviceSettings, {
      props: {
        capture: {
          ...capture,
          selectedDevice: null,
          devices: [],
          lastIssue: { code: 'no_input_device' as const, deviceName: null },
        },
      },
    })

    expect(wrapper.get('[data-testid="input-device-select"]').attributes('disabled')).toBeDefined()
    expect(wrapper.text()).toContain('未检测到输入设备')

    await wrapper.get('[data-testid="refresh-input-devices"]').trigger('click')

    expect(wrapper.emitted('refresh')).toEqual([[]])
  })

  test('reports a safe refresh error and locks controls while recording', () => {
    const wrapper = mount(InputDeviceSettings, {
      props: {
        refreshing: true,
        refreshError: 'native details must not be displayed',
        capture: {
          ...capture,
          status: 'recording',
        },
      },
    })

    expect(wrapper.text()).toContain('无法刷新输入设备')
    expect(wrapper.get('[data-testid="input-device-select"]').attributes('disabled')).toBeDefined()
    expect(wrapper.get('[data-testid="refresh-input-devices"]').attributes('disabled')).toBeDefined()
    expect(wrapper.text()).not.toContain('native details must not be displayed')
  })

  test('locks device selection while the native capture bridge is active', () => {
    const wrapper = mount(InputDeviceSettings, {
      props: {
        capture: {
          ...capture,
          bridge,
        },
      },
    })

    expect(wrapper.get('[data-testid="input-device-select"]').attributes('disabled')).toBeDefined()
    expect(wrapper.text()).toContain('录音期间不可切换输入设备')
  })
})
