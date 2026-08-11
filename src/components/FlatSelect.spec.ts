import { mount } from '@vue/test-utils'
import { describe, expect, test } from 'vitest'
import FlatSelect from './FlatSelect.vue'

const options = [
  { value: 'built-in', label: 'MacBook 麦克风' },
  { value: 'usb', label: 'USB 麦克风' },
]

describe('FlatSelect', () => {
  test('selects an option with pointer input and closes the listbox', async () => {
    const wrapper = mount(FlatSelect, {
      attachTo: document.body,
      props: { modelValue: 'built-in', options, label: '输入设备' },
    })

    await wrapper.get('[role="combobox"]').trigger('click')
    expect(wrapper.get('[role="combobox"]').attributes('aria-expanded')).toBe('true')
    await wrapper.get('[role="option"][data-value="usb"]').trigger('click')

    expect(wrapper.emitted('update:modelValue')).toEqual([['usb']])
    expect(wrapper.find('[role="listbox"]').exists()).toBe(false)
    wrapper.unmount()
  })

  test('supports keyboard navigation and selection', async () => {
    const wrapper = mount(FlatSelect, {
      attachTo: document.body,
      props: { modelValue: 'built-in', options, label: '输入设备' },
    })
    const trigger = wrapper.get('[role="combobox"]')

    await trigger.trigger('keydown', { key: 'ArrowDown' })
    await trigger.trigger('keydown', { key: 'ArrowDown' })
    await trigger.trigger('keydown', { key: 'Enter' })

    expect(wrapper.emitted('update:modelValue')).toEqual([['usb']])
    wrapper.unmount()
  })

  test('does not open while disabled', async () => {
    const wrapper = mount(FlatSelect, {
      props: { modelValue: '', options, label: '输入设备', disabled: true },
    })

    await wrapper.get('[role="combobox"]').trigger('click')

    expect(wrapper.find('[role="listbox"]').exists()).toBe(false)
    expect(wrapper.get('[role="combobox"]').attributes('disabled')).toBeDefined()
  })
})
