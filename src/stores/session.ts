import { defineStore } from 'pinia'
import { wordCovenantApi } from '@/lib/wordCovenantApi'
import type {
  AgentAction,
  CaptureInputKind,
  CaptureSession,
  TranscriptSpan,
} from '@/types'

export const useSessionStore = defineStore('session', {
  state: () => ({
    activeSession: null as CaptureSession | null,
    timeline: [] as TranscriptSpan[],
    actions: [] as AgentAction[],
    captureInput: 'microphone' as CaptureInputKind,
    isDevelopmentMockActive: false,
    isAdvancingDevelopmentMock: false,
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
          this.isDevelopmentMockActive = false
        } else if (this.captureInput === 'development_mock') {
          await this.startDevelopmentMockSession()
        } else {
          this.activeSession = await wordCovenantApi.startSession()
          this.timeline = await wordCovenantApi.listTimeline(this.activeSession.id)
        }
      } finally {
        this.isLoading = false
      }
    },

    setCaptureInput(input: CaptureInputKind) {
      if (!this.isRecording) {
        this.captureInput = input
      }
    },

    async startDevelopmentMockSession() {
      const session = await wordCovenantApi.startDevelopmentMockSession()
      this.activeSession = session
      this.timeline = []
      this.isDevelopmentMockActive = true
    },

    async advanceDevelopmentMock() {
      if (!this.isDevelopmentMockActive || this.isAdvancingDevelopmentMock) {
        return
      }

      this.isAdvancingDevelopmentMock = true
      try {
        const progress = await wordCovenantApi.advanceDevelopmentMock()
        if (progress.sessionId !== this.activeSession?.id) {
          return
        }
        this.mergeTimeline(progress.spans)
        if (progress.exhausted) {
          this.activeSession = await wordCovenantApi.stopSession()
          this.isDevelopmentMockActive = false
        }
      } catch (error) {
        this.isDevelopmentMockActive = false
        throw error
      } finally {
        this.isAdvancingDevelopmentMock = false
      }
    },

    mergeTimeline(spans: TranscriptSpan[]) {
      const byId = new Map(this.timeline.map((span) => [span.id, span]))
      for (const span of spans) {
        const current = byId.get(span.id)
        if (!current || span.revision >= current.revision) {
          byId.set(span.id, span)
        }
      }
      this.timeline = [...byId.values()].sort((left, right) => (
        left.captureStartNs - right.captureStartNs || left.revision - right.revision
      ))
    },

    async proposeLocalSpeech() {
      const action = await wordCovenantApi.proposeLocalSpeech()
      this.actions = [action, ...this.actions]
    },
  },
})
