import { defineStore } from 'pinia'
import { wordCovenantApi } from '@/lib/wordCovenantApi'
import type {
  AgentAction,
  CaptureInputKind,
  CaptureProjection,
  CaptureSession,
  TranscriptSpan,
} from '@/types'

const emptyCaptureProjection: CaptureProjection = {
  revision: -1,
  status: 'idle',
  permission: 'not_determined',
  selectedDevice: null,
  devices: [],
  meter: null,
  lastIssue: null,
}

export const useSessionStore = defineStore('session', {
  state: () => ({
    activeSession: null as CaptureSession | null,
    timeline: [] as TranscriptSpan[],
    actions: [] as AgentAction[],
    capture: { ...emptyCaptureProjection } as CaptureProjection,
    captureInput: 'microphone' as CaptureInputKind,
    isDevelopmentMockActive: false,
    isAdvancingDevelopmentMock: false,
    isLoading: false,
  }),

  getters: {
    isRecording: (state) => (
      state.capture.status === 'recording'
      || (state.isDevelopmentMockActive && state.activeSession?.state === 'recording')
    ),
    isAwaitingPermission: (state) => state.capture.status === 'awaiting_permission',
  },

  actions: {
    async initialize() {
      const [timeline, actions, capture] = await Promise.all([
        wordCovenantApi.listTimeline(),
        wordCovenantApi.listActions(),
        wordCovenantApi.getCaptureProjection(),
      ])
      this.timeline = timeline
      this.actions = actions
      this.applyCaptureProjection(capture)
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
          this.capture = {
            ...this.capture,
            status: 'awaiting_permission',
            lastIssue: null,
          }
          let session: CaptureSession
          try {
            session = await wordCovenantApi.startSession()
          } catch {
            try {
              this.applyCaptureProjection(await wordCovenantApi.getCaptureProjection())
              if (this.capture.status !== 'awaiting_permission') {
                return
              }
            } catch {
              // The fallback below makes a failed native start recoverable in the UI.
            }

            this.capture = {
              ...this.capture,
              revision: this.capture.revision + 1,
              status: 'failed',
              meter: null,
              lastIssue: {
                code: 'stream_start_failed',
                deviceName: this.capture.selectedDevice?.name ?? null,
              },
            }
            return
          }
          this.activeSession = session
          this.timeline = await wordCovenantApi.listTimeline(this.activeSession.id)
          this.applyCaptureProjection(await wordCovenantApi.getCaptureProjection())
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

    async selectInputDevice(deviceUid: string) {
      this.applyCaptureProjection(await wordCovenantApi.selectInputDevice(deviceUid))
    },

    applyCaptureProjection(projection: CaptureProjection) {
      if (projection.revision < this.capture.revision) {
        return
      }
      this.capture = projection
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
