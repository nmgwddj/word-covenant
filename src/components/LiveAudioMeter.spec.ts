import { mount } from '@vue/test-utils'
import { nextTick } from 'vue'
import { describe, expect, test } from 'vitest'
import type { CaptureMeter } from '@/types'
import LiveAudioMeter from './LiveAudioMeter.vue'

const meter: CaptureMeter = {
  rmsDbfs: -24,
  peakDbfs: -9,
  clipping: false,
  droppedPackets: 0,
}

function barHeights(wrapper: ReturnType<typeof mount>): string[] {
  return wrapper.findAll('.live-audio-meter__bar').map(bar => (bar.element as HTMLSpanElement).style.height)
}

describe('LiveAudioMeter', () => {
  test('keeps an inactive contour stable even when a previous meter value exists', () => {
    const wrapper = mount(LiveAudioMeter, {
      props: { active: false, meter },
    })

    expect(wrapper.attributes('data-state')).toBe('idle')
    expect(wrapper.attributes('aria-label')).toBe('输入声线未启用')
    expect(wrapper.text()).toContain('-- dBFS')
    expect(new Set(barHeights(wrapper))).toEqual(new Set(['12%']))
  })

  test('shows a stable waiting contour before the native meter has data', () => {
    const wrapper = mount(LiveAudioMeter, {
      props: { active: true, meter: null },
    })

    expect(wrapper.attributes('data-state')).toBe('waiting')
    expect(wrapper.attributes('aria-label')).toBe('输入声线正在等待输入电平')
    expect(new Set(barHeights(wrapper))).toEqual(new Set(['12%']))
  })

  test('draws a rolling input-level history from compact meter projections', async () => {
    const wrapper = mount(LiveAudioMeter, {
      props: { active: true, meter },
    })

    expect(wrapper.attributes('data-state')).toBe('active')
    expect(wrapper.attributes('aria-label')).toBe('实时输入电平声线，平均 -24 dBFS，峰值 -9 dBFS')
    expect(wrapper.text()).toContain('-9 dBFS')
    expect(new Set(barHeights(wrapper)).size).toBeGreaterThan(1)
    expect(wrapper.findAll('.live-audio-meter__bar')).toHaveLength(24)

    const priorLatestHeight = barHeights(wrapper).at(-1)
    await wrapper.setProps({
      meter: { ...meter, rmsDbfs: -48, peakDbfs: -40 },
    })
    await nextTick()

    expect(barHeights(wrapper).at(-1)).not.toBe(priorLatestHeight)
    expect(new Set(barHeights(wrapper)).size).toBeGreaterThan(2)
  })

  test('communicates clipping without leaking audio content', () => {
    const wrapper = mount(LiveAudioMeter, {
      props: {
        active: true,
        meter: { ...meter, clipping: true },
      },
    })

    expect(wrapper.attributes('data-state')).toBe('clipping')
    expect(wrapper.classes()).toContain('live-audio-meter--clipping')
    expect(wrapper.attributes('aria-label')).toContain('检测到削波')
  })

  test('bounds malformed or out-of-range dBFS values to the visual scale', () => {
    const wrapper = mount(LiveAudioMeter, {
      props: {
        active: true,
        meter: { rmsDbfs: -120, peakDbfs: 12, clipping: false, droppedPackets: 0 },
      },
    })

    for (const height of barHeights(wrapper)) {
      expect(Number.parseFloat(height)).toBeGreaterThanOrEqual(10)
      expect(Number.parseFloat(height)).toBeLessThanOrEqual(100)
    }
    expect(wrapper.text()).toContain('0 dBFS')
    expect(wrapper.attributes('aria-label')).not.toContain('12 dBFS')
  })
})
