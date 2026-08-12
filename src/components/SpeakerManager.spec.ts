import { mount } from '@vue/test-utils'
import { describe, expect, test } from 'vitest'
import type { TranscriptSpan } from '@/types'
import SpeakerManager from './SpeakerManager.vue'

const clusters = [
  {
    id: 'speaker-1',
    sessionId: 'session-one',
    label: '说话人 1',
    isUserNamed: false,
    labelRevision: 1,
    aliasRevision: 0,
    mergedIntoClusterId: null,
    canonicalClusterId: 'speaker-1',
    spanCount: 2,
    canEnrollVoiceProfile: true,
  },
  {
    id: 'speaker-2',
    sessionId: 'session-one',
    label: '说话人 2',
    isUserNamed: false,
    labelRevision: 1,
    aliasRevision: 0,
    mergedIntoClusterId: null,
    canonicalClusterId: 'speaker-2',
    spanCount: 1,
    canEnrollVoiceProfile: true,
  },
]

const spans: TranscriptSpan[] = [
  {
    id: 'span-1',
    sessionId: 'session-one',
    captureStartNs: 2_000_000_000,
    captureEndNs: 4_000_000_000,
    speakerClusterId: 'speaker-1',
    text: '请归类这条本地记录。',
    isFinal: true,
    revision: 4,
    source: 'local_inference' as const,
  },
]

function mountManager({
  error = null,
  managerClusters = clusters,
  managerSpans = spans,
  selectedSpanId = 'span-1',
  sessionStartNs = 0,
  useWallClock = false,
  attachTo,
}: {
  error?: string | null
  managerClusters?: typeof clusters
  managerSpans?: typeof spans
  selectedSpanId?: string | null
  sessionStartNs?: number
  useWallClock?: boolean
  attachTo?: HTMLElement
} = {}) {
  return mount(SpeakerManager, {
    attachTo,
    props: {
      clusters: managerClusters,
      spans: managerSpans,
      selectedSpanId,
      sessionStartNs,
      useWallClock,
      error,
    },
  })
}

describe('SpeakerManager', () => {
  test('submits a reassignment with the durable span revision', async () => {
    const wrapper = mountManager()

    expect(wrapper.get('aside').attributes('aria-modal')).toBeUndefined()
    expect(wrapper.text()).toContain('请归类这条本地记录。')
    await wrapper.get('[data-testid="speaker-target"]').trigger('click')
    await wrapper.get('[role="option"][data-value="speaker-2"]').trigger('click')
    await wrapper.get('.speaker-manager__assignment').trigger('submit')

    expect(wrapper.emitted('reassign')).toEqual([
      [
        {
          sessionId: 'session-one',
          logicalSpanId: 'span-1',
          expectedRevision: 4,
          targetClusterId: 'speaker-2',
        },
      ],
    ])
  })

  test('emits explicit create and rename requests without changing props optimistically', async () => {
    const wrapper = mountManager()

    await wrapper.get('[data-testid="create-speaker-cluster"]').trigger('click')
    await wrapper.get('[data-testid="speaker-label-speaker-2"]').setValue('主持人')
    await wrapper.findAll('.speaker-manager__row form')[1]!.trigger('submit')
    expect(wrapper.get('[role="alertdialog"]').text()).toContain('在本机记住这个声音')
    await wrapper.get('[data-testid="confirm-voice-enrollment"]').trigger('click')

    expect(wrapper.emitted('create')).toEqual([['session-one']])
    expect(wrapper.emitted('rename')).toEqual([
      [
        {
          sessionId: 'session-one',
          clusterId: 'speaker-2',
          expectedLabelRevision: 1,
          label: '主持人',
          consent: true,
        },
      ],
    ])
    expect(wrapper.props('clusters')).toEqual(clusters)
  })

  test('does not offer voice enrollment for an empty manually created cluster', async () => {
    const emptyCluster = { ...clusters[1]!, spanCount: 0, canEnrollVoiceProfile: false }
    const wrapper = mountManager({ managerClusters: [clusters[0]!, emptyCluster] })

    await wrapper.get('[data-testid="speaker-label-speaker-2"]').setValue('主持人')
    await wrapper.findAll('.speaker-manager__row form')[1]!.trigger('submit')

    expect(wrapper.find('[role="alertdialog"]').exists()).toBe(false)
    expect(wrapper.get('[role="alert"]').text()).toContain('还没有可用于声纹学习的录音')
    expect(wrapper.emitted('rename')).toBeUndefined()
  })

  test('exposes operation failures in an accessible alert', () => {
    const wrapper = mountManager({ error: '名称已被更新，请刷新后重试' })

    expect(wrapper.get('[role="alert"]').text()).toContain('名称已被更新')
    expect(wrapper.get('[data-testid="create-speaker-cluster"]').attributes('aria-label')).toBe('新增说话人归类')
  })

  test('uses the active session-relative capture timestamp', () => {
    const wrapper = mountManager({ sessionStartNs: 1_000_000_000 })

    expect(wrapper.get('.speaker-manager__selection-time').text()).toBe('00:01')
    expect(wrapper.get('.speaker-manager__selection-time').text()).not.toBe('00:02')
  })

  test('uses the persisted wall-clock capture timestamp after reopening a session', () => {
    const wallClockStart = '2026-08-08T02:03:04.000Z'
    const wrapper = mountManager({
      managerSpans: [{ ...spans[0]!, wallClockStart }],
      useWallClock: true,
    })
    const expectedWallClock = new Intl.DateTimeFormat('zh-CN', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    }).format(new Date(wallClockStart))

    expect(wrapper.get('.speaker-manager__selection-time').text()).toBe(expectedWallClock)
    expect(wrapper.get('.speaker-manager__selection-time').text()).not.toBe('00:02')
  })

  test('does not expose speaker correction controls for a partial transcript span', () => {
    const wrapper = mountManager({
      managerSpans: [{ ...spans[0]!, isFinal: false }],
    })

    expect(wrapper.get('.speaker-manager__empty').text()).toContain('未选择记录片段')
    expect(wrapper.find('[data-testid="speaker-target"]').exists()).toBe(false)
    expect(wrapper.find('[data-testid="create-speaker-cluster"]').exists()).toBe(false)
  })

  test('closes with Escape after the manager receives focus', async () => {
    const wrapper = mountManager({ attachTo: document.body })
    const manager = wrapper.get('aside')

    expect(document.activeElement).toBe(manager.element)
    await manager.trigger('keydown', { key: 'Escape' })

    expect(wrapper.emitted('close')).toEqual([[]])
    wrapper.unmount()
  })
})
