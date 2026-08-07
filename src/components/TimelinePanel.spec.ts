import { mount } from '@vue/test-utils'
import { describe, expect, test } from 'vitest'
import TimelinePanel from './TimelinePanel.vue'

describe('TimelinePanel', () => {
  test('renders capture timing and anonymous speaker labels', () => {
    const wrapper = mount(TimelinePanel, {
      props: {
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
})
