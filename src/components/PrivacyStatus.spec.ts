import { mount } from '@vue/test-utils'
import { describe, expect, test } from 'vitest'
import PrivacyStatus from './PrivacyStatus.vue'

describe('PrivacyStatus', () => {
  test('makes the local-only default visible', () => {
    const wrapper = mount(PrivacyStatus, {
      props: {
        status: {
          localOnly: true,
          egressEnabled: false,
          activeEgressApprovals: 0,
          recordingSessionId: null,
        },
      },
    })

    expect(wrapper.text()).toContain('本地模式')
    expect(wrapper.text()).not.toContain('受控出网')
  })
})
