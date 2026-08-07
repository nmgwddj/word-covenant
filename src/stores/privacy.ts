import { defineStore } from 'pinia'
import { wordCovenantApi } from '@/lib/wordCovenantApi'
import type { PrivacyStatus } from '@/types'

const initialStatus: PrivacyStatus = {
  localOnly: true,
  egressEnabled: false,
  activeEgressApprovals: 0,
  recordingSessionId: null,
}

export const usePrivacyStore = defineStore('privacy', {
  state: () => ({
    status: { ...initialStatus } as PrivacyStatus,
    isLoading: false,
    isUpdatingEgress: false,
  }),

  actions: {
    async refresh() {
      this.isLoading = true
      try {
        this.status = await wordCovenantApi.getPrivacyStatus()
      } finally {
        this.isLoading = false
      }
    },

    async setEgressEnabled(enabled: boolean) {
      this.isUpdatingEgress = true
      try {
        this.status = await wordCovenantApi.setEgressEnabled(enabled)
      } finally {
        this.isUpdatingEgress = false
      }
    },
  },
})
