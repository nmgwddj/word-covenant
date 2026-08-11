import { defineStore } from 'pinia'
import { wordCovenantApi } from '@/lib/wordCovenantApi'
import type { SpeechDetectionSettings } from '@/types'

export const defaultSpeechDetectionSettings: SpeechDetectionSettings = {
  rmsThresholdDbfs: -10,
}

const minimumRmsThresholdDbfs = -60
const maximumRmsThresholdDbfs = 0

function isValidThreshold(value: number): boolean {
  return Number.isInteger(value) && value >= minimumRmsThresholdDbfs && value <= maximumRmsThresholdDbfs
}

export const useSettingsStore = defineStore('settings', {
  state: () => ({
    speechDetection: { ...defaultSpeechDetectionSettings } as SpeechDetectionSettings,
    isLoadingSpeechDetection: false,
    isSavingSpeechDetection: false,
    speechDetectionError: null as string | null,
  }),

  actions: {
    clearSpeechDetectionError() {
      this.speechDetectionError = null
    },

    async initialize() {
      this.isLoadingSpeechDetection = true
      this.clearSpeechDetectionError()
      try {
        this.speechDetection = await wordCovenantApi.getSpeechDetectionSettings()
      } catch {
        this.speechDetectionError = '无法读取语音检测设置，已使用默认门限'
      } finally {
        this.isLoadingSpeechDetection = false
      }
    },

    async setRmsThresholdDbfs(rmsThresholdDbfs: number) {
      if (!isValidThreshold(rmsThresholdDbfs)) {
        this.speechDetectionError = '门限必须是 -60 到 0 之间的整数 dBFS'
        return null
      }
      if (this.isSavingSpeechDetection) return null

      this.isSavingSpeechDetection = true
      this.clearSpeechDetectionError()
      try {
        const saved = await wordCovenantApi.setSpeechDetectionSettings({ rmsThresholdDbfs })
        this.speechDetection = saved
        return saved
      } catch {
        this.speechDetectionError = '无法保存语音检测门限'
        return null
      } finally {
        this.isSavingSpeechDetection = false
      }
    },
  },
})
