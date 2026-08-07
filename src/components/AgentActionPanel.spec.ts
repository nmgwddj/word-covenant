import { mount } from '@vue/test-utils'
import { describe, expect, test } from 'vitest'
import AgentActionPanel from './AgentActionPanel.vue'

describe('AgentActionPanel', () => {
  test('keeps egress disabled and only emits a manual proposal request', async () => {
    const wrapper = mount(AgentActionPanel, {
      props: {
        actions: [],
        egressEnabled: false,
        activeEgressApprovals: 0,
      },
    })

    expect(wrapper.text()).toContain('会话出网访问')
    expect(wrapper.text()).toContain('已关闭')
    await wrapper.get('button').trigger('click')
    expect(wrapper.emitted('propose')).toHaveLength(1)
  })
})
