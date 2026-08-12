import { mount } from '@vue/test-utils'
import { describe, expect, test } from 'vitest'
import type { SessionSummary } from '@/types'
import SessionRail from './SessionRail.vue'

const sessions: SessionSummary[] = [
  {
    id: 'session-newest',
    startedAt: '2026-08-11T12:30:00.000Z',
    startedMonotonicNs: 2_000_000_000,
    stoppedAt: '2026-08-11T12:31:05.000Z',
    state: 'stopped',
    transcriptCount: 12,
  },
  {
    id: 'session-oldest',
    startedAt: '2026-08-10T10:00:00.000Z',
    startedMonotonicNs: 1_000_000_000,
    stoppedAt: null,
    state: 'recording',
    transcriptCount: 1,
  },
]

describe('SessionRail', () => {
  test('renders sessions in the given newest-first order with local time, duration, and transcript count', () => {
    const wrapper = mount(SessionRail, {
      props: {
        sessions,
        selectedSessionId: 'session-newest',
        recording: false,
      },
    })
    const items = wrapper.findAll('.session-item')

    expect(items.map(item => item.attributes('data-session-id'))).toEqual(['session-newest', 'session-oldest'])
    expect(items[0]?.get('time').text()).toMatch(/\d{4}年\d+月\d+日\s+\d{2}:\d{2}/)
    expect(items[0]?.text()).toContain('01:05')
    expect(items[0]?.text()).toContain('12 条转写')
    expect(items[1]?.text()).toContain('录音中')
    expect(items[1]?.text()).toContain('1 条转写')
    expect(items[0]?.attributes('aria-current')).toBe('true')
  })

  test('emits selection for a different session', async () => {
    const wrapper = mount(SessionRail, {
      props: {
        sessions,
        selectedSessionId: 'session-newest',
        recording: false,
      },
    })

    await wrapper.findAll('.session-item')[1]?.trigger('click')

    expect(wrapper.emitted('select')).toEqual([['session-oldest']])
  })

  test('disables every history item while recording', async () => {
    const wrapper = mount(SessionRail, {
      props: {
        sessions,
        selectedSessionId: 'session-oldest',
        recording: true,
      },
    })
    const items = wrapper.findAll('.session-item')

    expect(items.every(item => item.attributes('disabled') !== undefined)).toBe(true)
    expect(items[1]?.classes()).toContain('session-item--recording')
    await items[0]?.trigger('click')
    expect(wrapper.emitted('select')).toBeUndefined()
  })

  test('renders a quiet empty state without an inert create button', () => {
    const wrapper = mount(SessionRail, {
      props: {
        sessions: [],
        selectedSessionId: null,
        recording: false,
      },
    })

    expect(wrapper.text()).toContain('暂无本地会话')
    expect(wrapper.find('.session-item').exists()).toBe(false)
    expect(wrapper.find('button').exists()).toBe(false)
  })

  test('renders an archive error while retaining existing sessions', () => {
    const wrapper = mount(SessionRail, {
      props: {
        sessions,
        selectedSessionId: 'session-newest',
        recording: false,
        error: '读取会话历史失败',
      },
    })

    expect(wrapper.get('[role="alert"]').text()).toContain('读取会话历史失败')
    expect(wrapper.findAll('.session-item')).toHaveLength(2)
  })

  test('labels a stopped session without a stop timestamp as unfinished', () => {
    const wrapper = mount(SessionRail, {
      props: {
        sessions: [
          {
            ...sessions[0],
            stoppedAt: null,
          },
        ],
        selectedSessionId: 'session-newest',
        recording: false,
      },
    })

    expect(wrapper.text()).toContain('未正常结束')
    expect(wrapper.text()).not.toContain('录音中')
  })

  test('requires confirmation before emitting permanent deletion for a stopped session', async () => {
    const wrapper = mount(SessionRail, {
      attachTo: document.body,
      props: {
        sessions,
        selectedSessionId: 'session-newest',
        recording: false,
      },
    })

    expect(wrapper.findAll('.session-item__delete')).toHaveLength(1)
    expect(wrapper.find('.session-item__delete').attributes('aria-label')).toContain('删除')
    await wrapper.find('.session-item__delete').trigger('click')
    const dialog = document.body.querySelector<HTMLElement>('[role="alertdialog"]')
    expect(dialog?.textContent).toContain('永久删除')
    expect(wrapper.emitted('delete')).toBeUndefined()

    const confirm = dialog?.querySelectorAll<HTMLButtonElement>('button')[1]
    confirm?.click()
    await wrapper.vm.$nextTick()
    expect(wrapper.emitted('delete')).toEqual([['session-newest']])
    wrapper.unmount()
  })

  test('cancels deletion without emitting and disables deletion while recording', async () => {
    const wrapper = mount(SessionRail, {
      attachTo: document.body,
      props: {
        sessions,
        selectedSessionId: 'session-newest',
        recording: false,
      },
    })
    await wrapper.find('.session-item__delete').trigger('click')
    const cancel = document.body.querySelector<HTMLButtonElement>('.session-delete-dialog__cancel')
    cancel?.click()
    await wrapper.vm.$nextTick()
    expect(document.body.querySelector('[role="alertdialog"]')).toBeNull()
    expect(wrapper.emitted('delete')).toBeUndefined()

    await wrapper.setProps({ recording: true })
    expect(wrapper.find('.session-item__delete').attributes('disabled')).toBeDefined()
    wrapper.unmount()
  })
})
