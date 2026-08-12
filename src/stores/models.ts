import { defineStore } from 'pinia'
import { wordCovenantApi } from '@/lib/wordCovenantApi'
import type { ActiveLocalAsrProfile, BundledAsrStatus, LocalModelImportInput, RegisteredModel } from '@/types'

export const WHISPER_CPP_GGML_INPUT_FORMAT = 'whisper.cpp-ggml'

export function isWhisperCppAsrModel(model: RegisteredModel): boolean {
  return model.modelKind === 'speech_recognition' && model.inputFormat === WHISPER_CPP_GGML_INPUT_FORMAT
}

function unavailableBundledAsrStatus(): BundledAsrStatus {
  return {
    available: false,
    modelId: null,
    message: '内置离线转写模型不可用，请重新安装应用',
  }
}

function normalizeBundledAsrStatus(status: BundledAsrStatus): BundledAsrStatus {
  if (!status.available || !status.modelId) {
    return unavailableBundledAsrStatus()
  }

  return {
    available: true,
    modelId: status.modelId,
    message: null,
  }
}

export const useModelStore = defineStore('models', {
  state: () => ({
    models: [] as RegisteredModel[],
    activeAsrProfile: null as ActiveLocalAsrProfile | null,
    bundledAsrStatus: null as BundledAsrStatus | null,
    isLoading: false,
    isImporting: false,
    isSelectingActiveAsrModel: false,
    importError: null as string | null,
    activeAsrError: null as string | null,
  }),

  getters: {
    compatibleAsrModels: state =>
      state.models.filter(
        model =>
          isWhisperCppAsrModel(model) &&
          (model.id !== state.bundledAsrStatus?.modelId || state.bundledAsrStatus.available)
      ),
    bundledAsrModel: state => {
      const bundledModelId = state.bundledAsrStatus?.modelId
      if (!state.bundledAsrStatus?.available || !bundledModelId) return null
      return state.models.find(model => model.id === bundledModelId && isWhisperCppAsrModel(model)) ?? null
    },
    activeAsrModel: state => {
      const activeModelId = state.activeAsrProfile?.modelId
      if (!activeModelId) return null
      return (
        state.models.find(
          model =>
            model.id === activeModelId &&
            isWhisperCppAsrModel(model) &&
            (model.id !== state.bundledAsrStatus?.modelId || state.bundledAsrStatus.available)
        ) ?? null
      )
    },
    hasActiveCompatibleAsrModel: state => {
      const activeModelId = state.activeAsrProfile?.modelId
      return Boolean(
        activeModelId &&
        state.models.some(
          model =>
            model.id === activeModelId &&
            isWhisperCppAsrModel(model) &&
            (model.id !== state.bundledAsrStatus?.modelId || state.bundledAsrStatus.available)
        )
      )
    },
  },

  actions: {
    clearImportError() {
      this.importError = null
    },

    clearActiveAsrError() {
      this.activeAsrError = null
    },

    async initialize() {
      this.isLoading = true
      try {
        await this.refreshRuntimeState()
      } finally {
        this.isLoading = false
      }
    },

    async refreshRuntimeState() {
      const [models, activeAsrProfile, bundledAsrStatus] = await Promise.allSettled([
        wordCovenantApi.listLocalModels(),
        wordCovenantApi.getActiveLocalAsrProfile(),
        wordCovenantApi.getBundledAsrStatus(),
      ])

      if (models.status === 'fulfilled') {
        this.models = models.value
      }
      if (activeAsrProfile.status === 'fulfilled') {
        this.activeAsrProfile = activeAsrProfile.value
      }
      this.bundledAsrStatus =
        bundledAsrStatus.status === 'fulfilled'
          ? normalizeBundledAsrStatus(bundledAsrStatus.value)
          : unavailableBundledAsrStatus()
    },

    async selectLocalModelFile() {
      this.clearImportError()
      try {
        return await wordCovenantApi.selectLocalModelFile()
      } catch (error) {
        this.importError = error instanceof Error ? error.message : '无法选择本地模型文件'
        throw error
      }
    },

    async importLocalModel(input: LocalModelImportInput) {
      this.isImporting = true
      this.clearImportError()
      try {
        const model = await wordCovenantApi.importLocalModel(input)
        const models = new Map(this.models.map(existing => [existing.id, existing]))
        models.set(model.id, model)
        this.models = [...models.values()].sort((left, right) => right.importedAt.localeCompare(left.importedAt))
        return model
      } catch (error) {
        this.importError = error instanceof Error ? error.message : '无法导入本地模型'
        throw error
      } finally {
        this.isImporting = false
      }
    },

    async selectActiveLocalAsrModel(modelId: string) {
      this.clearActiveAsrError()
      const model = this.models.find(candidate => candidate.id === modelId)
      if (!model || !isWhisperCppAsrModel(model)) {
        const error = `请选择兼容 ${WHISPER_CPP_GGML_INPUT_FORMAT} 的本地语音识别模型`
        this.activeAsrError = error
        throw new Error(error)
      }

      this.isSelectingActiveAsrModel = true
      try {
        this.activeAsrProfile = await wordCovenantApi.selectActiveLocalAsrModel(modelId)
        return this.activeAsrProfile
      } catch (error) {
        this.activeAsrError = error instanceof Error ? error.message : '无法启用本地转写模型'
        throw error
      } finally {
        this.isSelectingActiveAsrModel = false
      }
    },
  },
})
