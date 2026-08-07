import { mount } from '@vue/test-utils'
import { describe, expect, test } from 'vitest'
import DevelopmentCaptureControl from './DevelopmentCaptureControl.vue'

describe('DevelopmentCaptureControl', () => {
  test('exposes a local development input selector without starting a session itself', async () => {
    const wrapper = mount(DevelopmentCaptureControl, {
      props: { selected: false },
    })

    const button = wrapper.get('[data-testid="development-capture-selector"]')
    expect(button.attributes('aria-pressed')).toBe('false')
    expect(button.attributes('title')).toBe('开发模拟音频输入')

    await button.trigger('click')

    expect(wrapper.emitted('select')).toHaveLength(1)
  })

  test('keeps the selected state visible and blocks changes while recording', async () => {
    const wrapper = mount(DevelopmentCaptureControl, {
      props: { selected: true, disabled: true },
    })

    const button = wrapper.get('[data-testid="development-capture-selector"]')
    expect(button.attributes('aria-pressed')).toBe('true')
    expect(button.attributes('disabled')).toBeDefined()

    await button.trigger('click')

    expect(wrapper.emitted('select')).toBeUndefined()
  })
})
