import { mount } from '@vue/test-utils'
import { describe, expect, test } from 'vitest'
import type { CaptureBridgeProjection } from '@/types'
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

const bridge: CaptureBridgeProjection = {
  status: 'armed',
  armed: true,
  shutdownRequested: false,
  workerFinished: false,
  metrics: {
    ingressPacketsConsumed: 18,
    ingressDiscontinuities: 0,
    segmenterFailures: 0,
    jobsAdmitted: 5,
    jobsCompleted: 4,
    jobQueueSaturated: 0,
    resultQueueSaturated: 0,
    unavailableEngineOutcomes: 4,
    engineFailureOutcomes: 0,
    shutdownOutcomes: 0,
    outcomeClaimsAborted: 0,
    jobQueueHighWatermark: 2,
    resultQueueHighWatermark: 1,
    pendingEventHighWatermark: 1,
    jobQueueDepth: 2,
    resultQueueDepth: 1,
    pendingEventDepth: 0,
    workerHoldsOutcome: false,
    ownedOutcomeLeaseActive: false,
    closing: false,
  },
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

  test('shows a local ASR selection requirement before microphone recording can begin', () => {
    const wrapper = mount(CaptureStatus, {
      props: {
        capture,
        asrModelReady: false,
      },
    })

    expect(wrapper.text()).toContain('请选择本地转写模型')
  })

  test('renders bounded bridge state and queue counts without transcript content', () => {
    const wrapper = mount(CaptureStatus, {
      props: {
        capture: {
          ...capture,
          bridge,
        },
      },
    })

    const summary = wrapper.get('[role="status"]')
    expect(summary.text()).toContain('桥接 运行')
    expect(summary.text()).toContain('任务2')
    expect(summary.text()).toContain('结果1')
    expect(summary.text()).toContain('待存0')
    expect(summary.text()).toContain('异常4')
    expect(summary.attributes('aria-label')).toContain('任务队列 2')
    expect(summary.attributes('aria-label')).toContain('结果队列 1')
    expect(summary.attributes('aria-label')).toContain('待持久化事件 0')
    expect(summary.attributes('aria-label')).toContain('没有结果等待持久化')
    expect(summary.attributes('aria-label')).toContain('本地引擎不可用 4')
  })

  test('distinguishes the bridge startup and drain phases from active recording', () => {
    const starting = mount(CaptureStatus, {
      props: {
        capture: {
          ...capture,
          status: 'awaiting_permission',
          bridge: { ...bridge, status: 'parked', armed: false },
        },
      },
    })

    expect(starting.text()).toContain('正在启动本地记录')
    expect(starting.get('[role="status"]').text()).toContain('桥接 启动中')
    expect(starting.get('select').attributes('disabled')).toBeDefined()

    const closing = mount(CaptureStatus, {
      props: {
        capture: {
          ...capture,
          status: 'recording',
          bridge: {
            ...bridge,
            status: 'closing',
            shutdownRequested: true,
            metrics: {
              ...bridge.metrics,
              pendingEventDepth: 3,
              segmenterFailures: 1,
              engineFailureOutcomes: 2,
              shutdownOutcomes: 1,
            },
          },
        },
      },
    })

    const summary = closing.get('[role="status"]')
    expect(closing.text()).toContain('正在完成本地记录')
    expect(summary.text()).toContain('桥接 收尾中')
    expect(summary.text()).toContain('待存3')
    expect(summary.text()).toContain('异常8')
    expect(summary.attributes('aria-label')).toContain('分段失败 1')
    expect(summary.attributes('aria-label')).toContain('本地引擎失败 2')
    expect(summary.attributes('aria-label')).toContain('收尾缺口 1')
  })

  test('caps visual counters while preserving their exact accessible values', () => {
    const wrapper = mount(CaptureStatus, {
      props: {
        capture: {
          ...capture,
          bridge: {
            ...bridge,
            metrics: {
              ...bridge.metrics,
              jobQueueDepth: 12_000,
            },
          },
        },
      },
    })

    const summary = wrapper.get('[role="status"]')
    expect(summary.text()).toContain('任务999+')
    expect(summary.attributes('aria-label')).toContain('任务队列 12000')
  })
})
