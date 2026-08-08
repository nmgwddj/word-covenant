import { defineStore } from 'pinia'
import { wordCovenantApi } from '@/lib/wordCovenantApi'
import type { LocalModelImportInput, RegisteredModel } from '@/types'

export const useModelStore = defineStore('models', {
  state: () => ({
    models: [] as RegisteredModel[],
    isLoading: false,
    isImporting: false,
    importError: null as string | null,
  }),

  actions: {
    clearImportError() {
      this.importError = null
    },

    async initialize() {
      this.isLoading = true
      try {
        this.models = await wordCovenantApi.listLocalModels()
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
  },
})
