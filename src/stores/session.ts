import { defineStore } from 'pinia'
import { wordCovenantApi } from '@/lib/wordCovenantApi'
import type { AgentAction, CaptureSession, TranscriptSpan } from '@/types'

export const useSessionStore = defineStore('session', {
  state: () => ({
    activeSession: null as CaptureSession | null,
    timeline: [] as TranscriptSpan[],
    actions: [] as AgentAction[],
    isLoading: false,
  }),

  getters: {
    isRecording: (state) => state.activeSession?.state === 'recording',
  },

  actions: {
    async initialize() {
      this.timeline = await wordCovenantApi.listTimeline()
      this.actions = await wordCovenantApi.listActions()
    },

    async toggleRecording() {
      this.isLoading = true
      try {
        if (this.isRecording) {
          this.activeSession = await wordCovenantApi.stopSession()
        } else {
          this.activeSession = await wordCovenantApi.startSession()
          this.timeline = await wordCovenantApi.listTimeline(this.activeSession.id)
        }
      } finally {
        this.isLoading = false
      }
    },

    async proposeLocalSpeech() {
      const action = await wordCovenantApi.proposeLocalSpeech()
      this.actions = [action, ...this.actions]
    },
  },
})
