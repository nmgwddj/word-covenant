import { mount } from '@vue/test-utils'
import { describe, expect, test } from 'vitest'
import RecordingControl from './RecordingControl.vue'

describe('RecordingControl', () => {
  test('emits a toggle request without inferring recording permission', async () => {
    const wrapper = mount(RecordingControl, {
      props: { recording: false },
    })

    await wrapper.get('button').trigger('click')

    expect(wrapper.emitted('toggle')).toHaveLength(1)
    expect(wrapper.text()).toContain('开始记录')
  })
})
