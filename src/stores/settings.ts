import { defineStore } from 'pinia'
import { wordCovenantApi } from '@/lib/wordCovenantApi'
import type { SpeechDetectionSettings, VoiceProfile } from '@/types'

export const defaultSpeechDetectionSettings: SpeechDetectionSettings = {
  mode: 'adaptive',
  rmsThresholdDbfs: -10,
}

const minimumRmsThresholdDbfs = -42
const maximumRmsThresholdDbfs = 0

function isValidThreshold(value: number): boolean {
  return Number.isInteger(value) && value >= minimumRmsThresholdDbfs && value <= maximumRmsThresholdDbfs
}

function isValidMode(value: string): value is SpeechDetectionSettings['mode'] {
  return value === 'adaptive' || value === 'manual'
}

export const useSettingsStore = defineStore('settings', {
  state: () => ({
    speechDetection: { ...defaultSpeechDetectionSettings } as SpeechDetectionSettings,
    isLoadingSpeechDetection: false,
    isSavingSpeechDetection: false,
    speechDetectionError: null as string | null,
    voiceProfiles: [] as VoiceProfile[],
    isLoadingVoiceProfiles: false,
    pendingVoiceProfileId: null as string | null,
    voiceProfileError: null as string | null,
  }),

  actions: {
    clearSpeechDetectionError() {
      this.speechDetectionError = null
    },

    clearVoiceProfileError() {
      this.voiceProfileError = null
    },

    async initialize() {
      this.isLoadingSpeechDetection = true
      this.clearSpeechDetectionError()
      const voiceProfiles = this.loadVoiceProfiles()
      try {
        this.speechDetection = await wordCovenantApi.getSpeechDetectionSettings()
      } catch {
        this.speechDetectionError = '无法读取语音检测设置，已使用默认门限'
      } finally {
        this.isLoadingSpeechDetection = false
      }
      await voiceProfiles
    },

    async setSpeechDetection(settings: SpeechDetectionSettings) {
      if (!isValidMode(settings.mode) || !isValidThreshold(settings.rmsThresholdDbfs)) {
        this.speechDetectionError = '门限必须是 -42 到 0 之间的整数 dBFS'
        return null
      }
      if (this.isSavingSpeechDetection) return null

      this.isSavingSpeechDetection = true
      this.clearSpeechDetectionError()
      try {
        const saved = await wordCovenantApi.setSpeechDetectionSettings(settings)
        this.speechDetection = saved
        return saved
      } catch {
        this.speechDetectionError = '无法保存语音检测门限'
        return null
      } finally {
        this.isSavingSpeechDetection = false
      }
    },

    async setRmsThresholdDbfs(rmsThresholdDbfs: number) {
      return this.setSpeechDetection({ mode: 'manual', rmsThresholdDbfs })
    },

    async loadVoiceProfiles() {
      this.isLoadingVoiceProfiles = true
      this.clearVoiceProfileError()
      try {
        this.voiceProfiles = await wordCovenantApi.listVoiceProfiles()
      } catch {
        this.voiceProfileError = '无法读取本机声纹档案'
      } finally {
        this.isLoadingVoiceProfiles = false
      }
    },

    async runVoiceProfileMutation(profileId: string, operation: () => Promise<VoiceProfile[]>) {
      if (this.pendingVoiceProfileId) return false
      this.pendingVoiceProfileId = profileId
      this.clearVoiceProfileError()
      try {
        this.voiceProfiles = await operation()
        return true
      } catch (error) {
        this.voiceProfileError = error instanceof Error ? error.message : '声纹档案操作未完成'
        return false
      } finally {
        this.pendingVoiceProfileId = null
      }
    },

    async renameVoiceProfile(profile: VoiceProfile, displayName: string) {
      return this.runVoiceProfileMutation(profile.id, () =>
        wordCovenantApi.renameVoiceProfile({
          profileId: profile.id,
          expectedRevision: profile.revision,
          displayName,
        })
      )
    },

    async relearnVoiceProfile(profile: VoiceProfile) {
      return this.runVoiceProfileMutation(profile.id, () =>
        wordCovenantApi.relearnVoiceProfile({ profileId: profile.id, expectedRevision: profile.revision })
      )
    },

    async addVoiceProfileConfirmedSample(profile: VoiceProfile) {
      return this.runVoiceProfileMutation(profile.id, () =>
        wordCovenantApi.addVoiceProfileConfirmedSample({
          profileId: profile.id,
          expectedRevision: profile.revision,
        })
      )
    },

    async deleteVoiceProfile(profile: VoiceProfile) {
      return this.runVoiceProfileMutation(profile.id, () =>
        wordCovenantApi.deleteVoiceProfile({ profileId: profile.id, expectedRevision: profile.revision })
      )
    },
  },
})
