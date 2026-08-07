import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, test, vi } from 'vitest'
import { wordCovenantApi } from '@/lib/wordCovenantApi'
import { usePrivacyStore } from './privacy'

describe('privacy store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.restoreAllMocks()
  })

  test('uses the policy command result when disabling outbound access', async () => {
    const returnedStatus = {
      localOnly: true,
      egressEnabled: false,
      activeEgressApprovals: 0,
      recordingSessionId: null,
    }
    const setEgressEnabled = vi.spyOn(wordCovenantApi, 'setEgressEnabled').mockResolvedValue(returnedStatus)
    const store = usePrivacyStore()

    await store.setEgressEnabled(false)

    expect(setEgressEnabled).toHaveBeenCalledWith(false)
    expect(store.status).toEqual(returnedStatus)
    expect(store.isUpdatingEgress).toBe(false)
  })
})
