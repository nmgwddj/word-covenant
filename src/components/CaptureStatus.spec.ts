import { mount } from '@vue/test-utils'
import { describe, expect, test } from 'vitest'
import CaptureStatus from './CaptureStatus.vue'

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

describe('CaptureStatus', () => {
  test('emits a stable input device UID when idle', async () => {
    const wrapper = mount(CaptureStatus, { props: { capture } })

    await wrapper.get('select').setValue('coreaudio:usb')

    expect(wrapper.emitted('select')).toEqual([['coreaudio:usb']])
  })

  test('locks device selection and exposes meter state while recording', () => {
    const wrapper = mount(CaptureStatus, {
      props: {
        capture: {
          ...capture,
          status: 'recording',
          meter: { rmsDbfs: -18, peakDbfs: -6, clipping: false, droppedPackets: 0 },
        },
      },
    })

    expect(wrapper.get('select').attributes('disabled')).toBeDefined()
    expect(wrapper.get('[role="meter"]').attributes('aria-label')).toContain('-6 dBFS')
    expect(wrapper.text()).toContain('麦克风记录中')
  })

  test('states a permission denial without claiming that recording continues', () => {
    const wrapper = mount(CaptureStatus, {
      props: {
        capture: {
          ...capture,
          status: 'failed',
          permission: 'denied',
          meter: null,
          lastIssue: { code: 'permission_denied', deviceName: null },
        },
      },
    })

    expect(wrapper.text()).toContain('麦克风权限被拒绝')
  })
})
