import { mount } from '@vue/test-utils'
import { describe, expect, test } from 'vitest'
import OutboundAccessControl from './OutboundAccessControl.vue'

function mountControl(props: Partial<InstanceType<typeof OutboundAccessControl>['$props']> = {}) {
  return mount(OutboundAccessControl, {
    props: {
      egressEnabled: false,
      profileApprovals: 0,
      ...props,
    },
  })
}

describe('OutboundAccessControl', () => {
  test('shows this session as outbound-disabled by default', () => {
    const wrapper = mountControl()

    expect(wrapper.text()).toContain('会话出网访问')
    expect(wrapper.text()).toContain('已关闭')
    expect(wrapper.text()).toContain('尚未批准命名配置，外部请求仍会被拦截')
    expect(wrapper.get('[data-testid="enable-egress"]').text()).toContain('启用出网访问')
    expect(wrapper.find('[role="dialog"]').exists()).toBe(false)
  })

  test('opens and cancels the explicit enable confirmation without emitting a change', async () => {
    const wrapper = mountControl()

    await wrapper.get('[data-testid="enable-egress"]').trigger('click')
    expect(wrapper.get('[role="dialog"]').text()).toContain('确认开启本会话出网访问？')

    await wrapper.get('[data-testid="cancel-enable-egress"]').trigger('click')
    expect(wrapper.find('[role="dialog"]').exists()).toBe(false)
    expect(wrapper.emitted('setEgressEnabled')).toBeUndefined()
  })

  test('emits an enable request only after confirmation', async () => {
    const wrapper = mountControl()

    await wrapper.get('[data-testid="enable-egress"]').trigger('click')
    await wrapper.get('[data-testid="confirm-enable-egress"]').trigger('click')

    expect(wrapper.find('[role="dialog"]').exists()).toBe(false)
    expect(wrapper.emitted('setEgressEnabled')).toEqual([[true]])
  })

  test('emits an immediate disable request when outbound access is enabled', async () => {
    const wrapper = mountControl({ egressEnabled: true })

    expect(wrapper.get('[data-testid="disable-egress"]').text()).toContain('立即关闭出网')
    await wrapper.get('[data-testid="disable-egress"]').trigger('click')

    expect(wrapper.emitted('setEgressEnabled')).toEqual([[false]])
  })
})
