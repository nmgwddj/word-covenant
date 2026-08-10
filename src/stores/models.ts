import { defineStore } from 'pinia'
import { wordCovenantApi } from '@/lib/wordCovenantApi'
import type { ActiveLocalAsrProfile, LocalModelImportInput, RegisteredModel } from '@/types'

export const WHISPER_CPP_GGML_INPUT_FORMAT = 'whisper.cpp-ggml'

export function isWhisperCppAsrModel(model: RegisteredModel): boolean {
  return model.modelKind === 'speech_recognition' && model.inputFormat === WHISPER_CPP_GGML_INPUT_FORMAT
}

export const useModelStore = defineStore('models', {
  state: () => ({
    models: [] as RegisteredModel[],
    activeAsrProfile: null as ActiveLocalAsrProfile | null,
    isLoading: false,
    isImporting: false,
    isSelectingActiveAsrModel: false,
    importError: null as string | null,
    activeAsrError: null as string | null,
  }),

  getters: {
    compatibleAsrModels: state => state.models.filter(isWhisperCppAsrModel),
    activeAsrModel: state => {
      const activeModelId = state.activeAsrProfile?.modelId
      if (!activeModelId) return null
      return state.models.find(model => model.id === activeModelId && isWhisperCppAsrModel(model)) ?? null
    },
    hasActiveCompatibleAsrModel: state => {
      const activeModelId = state.activeAsrProfile?.modelId
      return Boolean(
        activeModelId && state.models.some(model => model.id === activeModelId && isWhisperCppAsrModel(model))
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
        const [models, activeAsrProfile] = await Promise.all([
          wordCovenantApi.listLocalModels(),
          wordCovenantApi.getActiveLocalAsrProfile(),
        ])
        this.models = models
        this.activeAsrProfile = activeAsrProfile
      } finally {
        this.isLoading = false
      }
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
        const models = new Map(this.models.map((existing) => [existing.id, existing]))
        models.set(model.id, model)
        this.models = [...models.values()].sort((left, right) => (
          right.importedAt.localeCompare(left.importedAt)
        ))
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
