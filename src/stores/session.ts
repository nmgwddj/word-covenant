import { defineStore } from 'pinia'
import { wordCovenantApi } from '@/lib/wordCovenantApi'
import type {
  AgentAction,
  CaptureInputKind,
  CaptureProjection,
  CaptureSession,
  FinalTranscriptProjection,
  SpeakerCluster,
  SpeakerOperationResult,
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

function operationErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : '说话人归类操作未完成'
}

export const useSessionStore = defineStore('session', {
  state: () => ({
    activeSession: null as CaptureSession | null,
    timeline: [] as TranscriptSpan[],
    speakerClusters: [] as SpeakerCluster[],
    actions: [] as AgentAction[],
    capture: { ...emptyCaptureProjection } as CaptureProjection,
    captureInput: 'microphone' as CaptureInputKind,
    isDevelopmentMockActive: false,
    isAdvancingDevelopmentMock: false,
    isSpeakerOperationPending: false,
    speakerError: null as string | null,
    finalTranscriptProjectionRevisions: {} as Record<string, number>,
    isLoading: false,
  }),

  getters: {
    isRecording: state =>
      state.capture.status === 'recording' ||
      (state.isDevelopmentMockActive && state.activeSession?.state === 'recording'),
    isAwaitingPermission: state => state.capture.status === 'awaiting_permission',
  },

  actions: {
    async initialize() {
      const [timeline, actions, capture, speakerClusters] = await Promise.all([
        wordCovenantApi.listTimeline(),
        wordCovenantApi.listActions(),
        wordCovenantApi.getCaptureProjection(),
        wordCovenantApi.listSpeakerClusters(),
      ])
      this.timeline = timeline
      this.speakerClusters = speakerClusters
      this.actions = actions
      this.applyCaptureProjection(capture)
    },

    async toggleRecording() {
      this.isLoading = true
      try {
        if (this.isRecording) {
          const stoppedSession = await wordCovenantApi.stopSession()
          this.activeSession = stoppedSession
          this.isDevelopmentMockActive = false
          if (this.captureInput === 'microphone' && stoppedSession) {
            await this.refreshTimelineForCurrentSession(stoppedSession.id)
          }
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
          const [timeline, speakerClusters, capture] = await Promise.all([
            wordCovenantApi.listTimeline(this.activeSession.id),
            wordCovenantApi.listSpeakerClusters(this.activeSession.id),
            wordCovenantApi.getCaptureProjection(),
          ])
          this.timeline = timeline
          this.speakerClusters = speakerClusters
          this.applyCaptureProjection(capture)
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
      this.speakerClusters = await wordCovenantApi.listSpeakerClusters(session.id)
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
      const byId = new Map(this.timeline.map(span => [span.id, span]))
      for (const span of spans) {
        const current = byId.get(span.id)
        if (!current || span.revision >= current.revision) {
          byId.set(span.id, span)
        }
      }
      this.timeline = [...byId.values()].sort(
        (left, right) => left.captureStartNs - right.captureStartNs || left.revision - right.revision
      )
      this.syncSpeakerSpanCounts()
    },

    syncSpeakerSpanCounts() {
      const spansByCluster = new Map<string, number>()
      for (const span of this.timeline) {
        if (span.speakerClusterId) {
          spansByCluster.set(span.speakerClusterId, (spansByCluster.get(span.speakerClusterId) ?? 0) + 1)
        }
      }
      this.speakerClusters = this.speakerClusters.map(cluster => ({
        ...cluster,
        spanCount: spansByCluster.get(cluster.id) ?? 0,
      }))
    },

    replaceTimelineForSession(sessionId: string, spans: TranscriptSpan[]) {
      const otherSessions = this.timeline.filter(span => span.sessionId !== sessionId)
      this.timeline = [...otherSessions, ...spans].sort(
        (left, right) => left.captureStartNs - right.captureStartNs || left.revision - right.revision
      )
    },

    async applyFinalTranscriptProjection(projection: FinalTranscriptProjection) {
      const previousRevision = this.finalTranscriptProjectionRevisions[projection.sessionId] ?? -1
      if (projection.revision <= previousRevision) {
        return
      }
      this.finalTranscriptProjectionRevisions[projection.sessionId] = projection.revision

      if (this.activeSession?.id !== projection.sessionId) {
        return
      }

      const timeline = await wordCovenantApi.listTimeline(projection.sessionId)
      if (
        this.activeSession?.id !== projection.sessionId ||
        this.finalTranscriptProjectionRevisions[projection.sessionId] !== projection.revision
      ) {
        return
      }
      this.replaceTimelineForSession(projection.sessionId, timeline)
    },

    async refreshTimelineForCurrentSession(sessionId: string) {
      const timeline = await wordCovenantApi.listTimeline(sessionId)
      if (this.activeSession?.id === sessionId) {
        this.replaceTimelineForSession(sessionId, timeline)
      }
    },

    async applySpeakerOperationResult(sessionId: string, result: SpeakerOperationResult) {
      if (result.updatedSpans.length > 0) {
        const timeline = await wordCovenantApi.listTimeline(sessionId)
        this.replaceTimelineForSession(sessionId, timeline)
      }
      this.speakerClusters = result.clusters
    },

    async refreshSpeakerProjections(sessionId: string) {
      const [timeline, speakerClusters] = await Promise.all([
        wordCovenantApi.listTimeline(sessionId),
        wordCovenantApi.listSpeakerClusters(sessionId),
      ])
      this.replaceTimelineForSession(sessionId, timeline)
      this.speakerClusters = speakerClusters
    },

    async runSpeakerOperation(
      sessionId: string,
      operation: () => Promise<SpeakerOperationResult>
    ): Promise<SpeakerOperationResult | null> {
      if (this.isSpeakerOperationPending) return null

      this.isSpeakerOperationPending = true
      this.speakerError = null
      let operationCommitted = false
      try {
        const result = await operation()
        operationCommitted = true
        await this.applySpeakerOperationResult(sessionId, result)
        return result
      } catch (error) {
        if (!operationCommitted) {
          try {
            await this.refreshSpeakerProjections(sessionId)
          } catch {
            // Keep the original operation error: a failed refresh cannot make a rejected request succeed.
          }
          this.speakerError = operationErrorMessage(error)
        } else {
          this.speakerError = '说话人归类已保存，但本地记录刷新失败，请重新打开后核对'
        }
        return null
      } finally {
        this.isSpeakerOperationPending = false
      }
    },

    async createSpeakerCluster(sessionId: string) {
      return this.runSpeakerOperation(sessionId, () =>
        wordCovenantApi.createSpeakerCluster({
          sessionId,
        })
      )
    },

    async renameSpeakerCluster(input: {
      sessionId: string
      clusterId: string
      expectedLabelRevision: number
      label: string
    }) {
      return this.runSpeakerOperation(input.sessionId, () => wordCovenantApi.renameSpeakerCluster(input))
    },

    async reassignTranscriptSpeaker(input: {
      sessionId: string
      logicalSpanId: string
      expectedRevision: number
      targetClusterId: string | null
    }) {
      return this.runSpeakerOperation(input.sessionId, () => wordCovenantApi.reassignTranscriptSpeaker(input))
    },

    clearSpeakerError() {
      this.speakerError = null
    },

    async proposeLocalSpeech() {
      const action = await wordCovenantApi.proposeLocalSpeech()
      this.actions = [action, ...this.actions]
    },
  },
})
