import { mount } from '@vue/test-utils'
import { describe, expect, test } from 'vitest'
import type { VoiceProfile } from '@/types'
import VoiceProfileSettings from './VoiceProfileSettings.vue'

const profile: VoiceProfile = {
  id: 'profile-one',
  revision: 3,
  displayName: '主持人',
  state: 'learning',
  confirmedDurationNs: 2_500_000_000,
  readyConfirmedDurationNs: 4_000_000_000,
  modelId: 'campplus',
  modelVersion: '2024-10-14',
  lastConfirmationAt: '2026-08-12T08:00:00.000Z',
  canAddConfirmedSample: true,
  updatedAt: '2026-08-12T08:00:00.000Z',
}

describe('VoiceProfileSettings', () => {
  test('shows local learning progress and emits explicit management actions', async () => {
    const wrapper = mount(VoiceProfileSettings, { props: { profiles: [profile] } })

    expect(wrapper.text()).toContain('学习中')
    expect(wrapper.text()).toContain('2.5 秒 / 4.0 秒')
    expect(wrapper.text()).toContain('2024-10-14')

    await wrapper.get('input').setValue('会议主持人')
    await wrapper.get('form').trigger('submit')
    await wrapper.findAll('.voice-profile__actions button')[0]!.trigger('click')
    await wrapper.findAll('.voice-profile__actions button')[1]!.trigger('click')

    expect(wrapper.emitted('rename')).toEqual([[profile, '会议主持人']])
    expect(wrapper.emitted('addSample')).toEqual([[profile]])
    expect(wrapper.emitted('relearn')).toEqual([[profile]])
  })

  test('requires confirmation and explains historical label retention before deletion', async () => {
    const wrapper = mount(VoiceProfileSettings, { props: { profiles: [profile] } })

    await wrapper.findAll('.voice-profile__actions button')[2]!.trigger('click')

    expect(wrapper.get('[role="alertdialog"]').text()).toContain('历史转写中的姓名快照仍会保留')
    await wrapper.get('[data-testid="confirm-delete-voice-profile"]').trigger('click')
    expect(wrapper.emitted('delete')).toEqual([[profile]])
  })
})
