import { mount } from '@vue/test-utils'
import { describe, expect, test } from 'vitest'
import TimelinePanel from './TimelinePanel.vue'

describe('TimelinePanel', () => {
  test('renders the newest speech first without mutating the session timeline', () => {
    const spans = [
      {
        id: 'oldest',
        sessionId: 'session-one',
        captureStartNs: 1_000_000_000,
        captureEndNs: 2_000_000_000,
        speakerClusterId: null,
        text: '最早内容',
        isFinal: true,
        revision: 1,
        source: 'local_inference' as const,
      },
      {
        id: 'latest-short',
        sessionId: 'session-one',
        captureStartNs: 5_000_000_000,
        captureEndNs: 6_000_000_000,
        speakerClusterId: null,
        text: '同一时刻较短内容',
        isFinal: true,
        revision: 1,
        source: 'local_inference' as const,
      },
      {
        id: 'middle',
        sessionId: 'session-one',
        captureStartNs: 3_000_000_000,
        captureEndNs: 4_000_000_000,
        speakerClusterId: null,
        text: '中间内容',
        isFinal: true,
        revision: 1,
        source: 'local_inference' as const,
      },
      {
        id: 'latest-long',
        sessionId: 'session-one',
        captureStartNs: 5_000_000_000,
        captureEndNs: 7_000_000_000,
        speakerClusterId: null,
        text: '最新内容',
        isFinal: false,
        revision: 2,
        source: 'local_inference' as const,
      },
    ]
    const originalOrder = spans.map(span => span.id)
    const wrapper = mount(TimelinePanel, { props: { spans } })

    expect(wrapper.findAll('.timeline-entry').map(entry => entry.find('p').text())).toEqual([
      '最新内容',
      '同一时刻较短内容',
      '中间内容',
      '最早内容',
    ])
    expect(spans.map(span => span.id)).toEqual(originalOrder)
    expect(wrapper.findAll('.timeline-entry')[0]?.text()).toContain('转写中')
  })

  test('renders capture timing and anonymous speaker labels', () => {
    const wrapper = mount(TimelinePanel, {
      props: {
        speakerClusters: [
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
          },
        ],
        spans: [
          {
            id: 'one',
            sessionId: 'session-one',
            captureStartNs: 65_000_000_000,
            captureEndNs: 67_000_000_000,
            speakerClusterId: 'speaker-2',
            text: '本地时间线记录',
            isFinal: true,
            revision: 1,
            source: 'synthetic',
          },
        ],
      },
    })

    expect(wrapper.text()).toContain('01:05')
    expect(wrapper.text()).toContain('说话人 2')
    expect(wrapper.text()).toContain('本地时间线记录')
  })

  test('renders an unavailable catalog reference as unclassified', () => {
    const wrapper = mount(TimelinePanel, {
      props: {
        spans: [
          {
            id: 'missing-cluster',
            sessionId: 'session-one',
            captureStartNs: 0,
            captureEndNs: 1_000_000_000,
            speakerClusterId: 'speaker-missing',
            text: '目录中没有这个归类。',
            isFinal: true,
            revision: 1,
            source: 'local_inference',
          },
        ],
      },
    })

    expect(wrapper.text()).toContain('未归类')
    expect(wrapper.text()).not.toContain('说话人 missing')
  })

  test('uses a durable speaker label and opens the compact manager from a timeline entry', async () => {
    const wrapper = mount(TimelinePanel, {
      props: {
        speakerClusters: [
          {
            id: 'speaker-2',
            sessionId: 'session-one',
            label: '主持人',
            isUserNamed: true,
            labelRevision: 3,
            aliasRevision: 0,
            mergedIntoClusterId: null,
            canonicalClusterId: 'speaker-2',
            spanCount: 1,
          },
        ],
        spans: [
          {
            id: 'one',
            sessionId: 'session-one',
            captureStartNs: 0,
            captureEndNs: 2_000_000_000,
            speakerClusterId: 'speaker-2',
            text: '可校对的本地记录',
            isFinal: true,
            revision: 4,
            source: 'local_inference',
          },
        ],
      },
    })

    expect(wrapper.text()).toContain('主持人')
    await wrapper.get('[data-testid="open-speaker-manager"]').trigger('click')
    expect(wrapper.emitted('openSpeakerManager')).toEqual([['one']])
  })

  test('does not expose speaker management for a partial transcript span', () => {
    const wrapper = mount(TimelinePanel, {
      props: {
        speakerClusters: [
          {
            id: 'speaker-2',
            sessionId: 'session-one',
            label: '主持人',
            isUserNamed: true,
            labelRevision: 3,
            aliasRevision: 0,
            mergedIntoClusterId: null,
            canonicalClusterId: 'speaker-2',
            spanCount: 1,
          },
        ],
        spans: [
          {
            id: 'partial',
            sessionId: 'session-one',
            captureStartNs: 0,
            captureEndNs: 2_000_000_000,
            speakerClusterId: 'speaker-2',
            text: '仍在本地转写中的记录',
            isFinal: false,
            revision: 1,
            source: 'local_inference',
          },
        ],
      },
    })

    expect(wrapper.find('[data-testid="open-speaker-manager"]').exists()).toBe(false)
    expect(wrapper.get('.speaker-tag').element.tagName).toBe('SPAN')
    expect(wrapper.find('button.speaker-tag').exists()).toBe(false)
    expect(wrapper.text()).toContain('转写中')
  })

  test('renders an active session timeline relative to its monotonic start', () => {
    const wrapper = mount(TimelinePanel, {
      props: {
        sessionStartNs: 65_000_000_000,
        spans: [
          {
            id: 'relative',
            sessionId: 'session-one',
            captureStartNs: 66_000_000_000,
            captureEndNs: 68_000_000_000,
            speakerClusterId: 'speaker-1',
            text: '会话相对时间',
            isFinal: true,
            revision: 1,
            source: 'synthetic',
          },
        ],
      },
    })

    expect(wrapper.text()).toContain('00:01')
    expect(wrapper.text()).not.toContain('01:06')
  })

  test('renders a persisted wall-clock timestamp after reopening without a live session', () => {
    const wrapper = mount(TimelinePanel, {
      props: {
        useWallClock: true,
        spans: [
          {
            id: 'archived',
            sessionId: 'session-one',
            captureStartNs: 66_000_000_000,
            captureEndNs: 68_000_000_000,
            wallClockStart: '2026-08-08T02:03:04.000Z',
            speakerClusterId: 'speaker-1',
            text: '已归档会话',
            isFinal: true,
            revision: 1,
            source: 'local_inference',
          },
        ],
      },
    })

    const expectedWallClock = new Intl.DateTimeFormat('zh-CN', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    }).format(new Date('2026-08-08T02:03:04.000Z'))

    expect(wrapper.find('time').attributes('datetime')).toBe('2026-08-08T02:03:04.000Z')
    expect(wrapper.find('time').text()).toBe(expectedWallClock)
    expect(wrapper.text()).not.toContain('01:06')
  })
})
