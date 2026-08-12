import { defineStore } from 'pinia'
import { wordCovenantApi } from '@/lib/wordCovenantApi'
import { useModelStore } from '@/stores/models'
import type {
  AgentAction,
  CaptureInputKind,
  CaptureProjection,
  CaptureSession,
  FinalTranscriptProjection,
  SessionSummary,
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
  const message =
    error instanceof Error
      ? error.message
      : typeof error === 'string'
        ? error
        : typeof error === 'object' && error !== null && 'message' in error && typeof error.message === 'string'
          ? error.message
          : ''
  const normalized = message.trim()
  if (normalized.includes('no high-quality local sample available for enrollment')) {
    return '该归类还没有可用于声纹学习的录音，请重新录制一段清晰人声后再命名'
  }
  if (normalized.includes('explicit local voice profile consent is required')) {
    return '记住声纹前需要获得你的明确同意'
  }
  return normalized || '说话人归类操作未完成'
}

function sessionHistoryErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : '会话历史加载失败'
}

function sessionDeletionErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : '会话删除失败'
}

function sessionSummaryFromCapture(session: CaptureSession, transcriptCount: number): SessionSummary {
  return {
    ...session,
    transcriptCount,
  }
}

export const useSessionStore = defineStore('session', {
  state: () => ({
    activeSession: null as CaptureSession | null,
    sessions: [] as SessionSummary[],
    selectedSessionId: null as string | null,
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
    isSessionHistoryLoading: false,
    deletingSessionId: null as string | null,
    sessionHistoryError: null as string | null,
  }),

  getters: {
    isRecording: state =>
      state.capture.status === 'recording' ||
      (state.isDevelopmentMockActive && state.activeSession?.state === 'recording'),
    isAwaitingPermission: state => state.capture.status === 'awaiting_permission',
    selectedSession: state => state.sessions.find(session => session.id === state.selectedSessionId) ?? null,
  },

  actions: {
    async initialize() {
      const historyInitialization = this.initializeSessionHistory()
      const [actions, capture] = await Promise.all([
        wordCovenantApi.listActions(),
        wordCovenantApi.getCaptureProjection(),
      ])
      this.actions = actions
      this.applyCaptureProjection(capture)
      await historyInitialization
    },

    async initializeSessionHistory() {
      this.isSessionHistoryLoading = true
      this.sessionHistoryError = null
      try {
        const sessions = await wordCovenantApi.listSessions()
        this.sessions = sessions
        if (sessions.length === 0) {
          this.selectedSessionId = null
          this.timeline = []
          this.speakerClusters = []
          return
        }

        const newestSessionId = sessions[0]!.id
        const [timeline, speakerClusters] = await Promise.all([
          wordCovenantApi.listTimeline(newestSessionId),
          wordCovenantApi.listSpeakerClusters(newestSessionId),
        ])
        this.selectedSessionId = newestSessionId
        this.timeline = timeline
        this.speakerClusters = speakerClusters
      } catch (error) {
        this.sessionHistoryError = sessionHistoryErrorMessage(error)
      } finally {
        this.isSessionHistoryLoading = false
      }
    },

    async refreshSessionHistory(): Promise<boolean> {
      this.isSessionHistoryLoading = true
      this.sessionHistoryError = null
      try {
        this.sessions = await wordCovenantApi.listSessions()
        return true
      } catch (error) {
        this.sessionHistoryError = sessionHistoryErrorMessage(error)
        return false
      } finally {
        this.isSessionHistoryLoading = false
      }
    },

    async selectSession(sessionId: string): Promise<boolean> {
      if (this.isRecording || this.isLoading || this.deletingSessionId || sessionId === this.selectedSessionId) {
        return sessionId === this.selectedSessionId && !this.isRecording && !this.isLoading && !this.deletingSessionId
      }
      if (!this.sessions.some(session => session.id === sessionId)) {
        this.sessionHistoryError = '会话不存在或已不可用'
        return false
      }

      this.isSessionHistoryLoading = true
      this.sessionHistoryError = null
      try {
        const [timeline, speakerClusters] = await Promise.all([
          wordCovenantApi.listTimeline(sessionId),
          wordCovenantApi.listSpeakerClusters(sessionId),
        ])
        this.selectedSessionId = sessionId
        this.timeline = timeline
        this.speakerClusters = speakerClusters
        return true
      } catch (error) {
        this.sessionHistoryError = sessionHistoryErrorMessage(error)
        return false
      } finally {
        this.isSessionHistoryLoading = false
      }
    },

    async deleteSession(sessionId: string): Promise<boolean> {
      if (this.isRecording || this.isLoading || this.isSessionHistoryLoading || this.deletingSessionId) {
        return false
      }
      const deletedIndex = this.sessions.findIndex(session => session.id === sessionId)
      if (deletedIndex < 0) {
        this.sessionHistoryError = '会话不存在或已删除'
        return false
      }
      if (this.sessions[deletedIndex]?.state !== 'stopped') {
        this.sessionHistoryError = '录音中的会话不能删除，请先停止录音'
        return false
      }

      this.deletingSessionId = sessionId
      this.sessionHistoryError = null
      try {
        await wordCovenantApi.deleteSession(sessionId)
        const remaining = this.sessions.filter(session => session.id !== sessionId)
        this.sessions = remaining
        delete this.finalTranscriptProjectionRevisions[sessionId]
        if (this.selectedSessionId !== sessionId) return true

        const replacement = remaining[Math.min(deletedIndex, remaining.length - 1)]
        if (!replacement) {
          this.selectedSessionId = null
          this.timeline = []
          this.speakerClusters = []
          return true
        }

        const [timeline, speakerClusters] = await Promise.all([
          wordCovenantApi.listTimeline(replacement.id),
          wordCovenantApi.listSpeakerClusters(replacement.id),
        ])
        this.selectedSessionId = replacement.id
        this.timeline = timeline
        this.speakerClusters = speakerClusters
        return true
      } catch (error) {
        if (!this.sessions.some(session => session.id === sessionId)) {
          this.selectedSessionId = null
          this.timeline = []
          this.speakerClusters = []
          this.sessionHistoryError = '会话已删除，但相邻记录加载失败，请重新打开后核对'
        } else {
          this.sessionHistoryError = sessionDeletionErrorMessage(error)
        }
        return false
      } finally {
        this.deletingSessionId = null
      }
    },

    async toggleRecording() {
      this.isLoading = true
      try {
        if (this.isRecording) {
          const stoppedSession = await wordCovenantApi.stopSession()
          this.activeSession = stoppedSession
          this.isDevelopmentMockActive = false
          if (stoppedSession) {
            this.selectedSessionId = stoppedSession.id
            await this.refreshTimelineForCurrentSession(stoppedSession.id)
            this.upsertSessionSummary(stoppedSession)
          }
          await this.refreshSessionHistory()
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
            let startProjection: CaptureProjection | null = null
            try {
              startProjection = await wordCovenantApi.getCaptureProjection()
              this.applyCaptureProjection(startProjection)
            } catch {
              // The fallback below makes a failed native start recoverable in the UI.
            }

            if (startProjection?.status !== 'recording') {
              if (this.capture.status !== 'failed' && this.capture.status !== 'interrupted') {
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
              }

              // Native model verification happens immediately before microphone
              // activation. Refresh its compact, path-free projection after a
              // rejected start so an invalid bundled model is visible to users.
              await useModelStore().refreshRuntimeState()
            }
            return
          }
          this.activeSession = session
          this.selectedSessionId = session.id
          this.timeline = []
          this.speakerClusters = []
          this.upsertSessionSummary(session, 0)
          const [timeline, speakerClusters, capture] = await Promise.all([
            wordCovenantApi.listTimeline(session.id),
            wordCovenantApi.listSpeakerClusters(session.id),
            wordCovenantApi.getCaptureProjection(),
          ])
          this.timeline = timeline
          this.speakerClusters = speakerClusters
          this.applyCaptureProjection(capture)
          await this.refreshSessionHistory()
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
      this.selectedSessionId = session.id
      this.timeline = []
      this.speakerClusters = await wordCovenantApi.listSpeakerClusters(session.id)
      this.isDevelopmentMockActive = true
      this.upsertSessionSummary(session, 0)
      await this.refreshSessionHistory()
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
          if (this.activeSession) {
            this.selectedSessionId = this.activeSession.id
            await this.refreshTimelineForCurrentSession(this.activeSession.id)
            this.upsertSessionSummary(this.activeSession)
          }
          await this.refreshSessionHistory()
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
      this.syncSelectedSessionTranscriptCount()
    },

    upsertSessionSummary(session: CaptureSession, transcriptCount?: number) {
      const current = this.sessions.find(summary => summary.id === session.id)
      const count =
        transcriptCount ??
        (this.selectedSessionId === session.id
          ? this.selectedTimelineTranscriptCount()
          : (current?.transcriptCount ?? 0))
      const summary = sessionSummaryFromCapture(session, count)
      this.sessions = [summary, ...this.sessions.filter(item => item.id !== session.id)].sort(
        (left, right) => Date.parse(right.startedAt) - Date.parse(left.startedAt) || right.id.localeCompare(left.id)
      )
    },

    selectedTimelineTranscriptCount(): number {
      if (!this.selectedSessionId) return 0
      return this.timeline.filter(span => span.sessionId === this.selectedSessionId && span.isFinal).length
    },

    syncSelectedSessionTranscriptCount() {
      if (!this.selectedSessionId) return
      const transcriptCount = this.selectedTimelineTranscriptCount()
      this.sessions = this.sessions.map(session =>
        session.id === this.selectedSessionId ? { ...session, transcriptCount } : session
      )
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
      if (this.selectedSessionId !== sessionId) return
      this.timeline = [...spans].sort(
        (left, right) => left.captureStartNs - right.captureStartNs || left.revision - right.revision
      )
      this.syncSelectedSessionTranscriptCount()
    },

    async applyFinalTranscriptProjection(projection: FinalTranscriptProjection) {
      const previousRevision = this.finalTranscriptProjectionRevisions[projection.sessionId] ?? -1
      if (projection.revision <= previousRevision) {
        return
      }
      this.finalTranscriptProjectionRevisions[projection.sessionId] = projection.revision

      if (this.selectedSessionId !== projection.sessionId) {
        return
      }

      const [timeline, speakerClusters] = await Promise.all([
        wordCovenantApi.listTimeline(projection.sessionId),
        wordCovenantApi.listSpeakerClusters(projection.sessionId),
      ])
      if (
        this.selectedSessionId !== projection.sessionId ||
        this.finalTranscriptProjectionRevisions[projection.sessionId] !== projection.revision
      ) {
        return
      }
      this.replaceTimelineForSession(projection.sessionId, timeline)
      this.speakerClusters = speakerClusters
    },

    async refreshTimelineForCurrentSession(sessionId: string) {
      const timeline = await wordCovenantApi.listTimeline(sessionId)
      if (this.selectedSessionId === sessionId) {
        this.replaceTimelineForSession(sessionId, timeline)
      }
    },

    async applySpeakerOperationResult(sessionId: string, result: SpeakerOperationResult) {
      if (this.selectedSessionId !== sessionId) return
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
      if (this.selectedSessionId !== sessionId) return
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
      consent: boolean
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
