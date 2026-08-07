import packageJson from '../../package.json'
import tauriConfig from '../../src-tauri/tauri.conf.json'
import capability from '../../src-tauri/capabilities/default.json'
import { describe, expect, test } from 'vitest'

describe('WordCovenant product metadata', () => {
  test('brands the application and disables shell capabilities by default', () => {
    expect(packageJson.name).toBe('word-covenant')
    expect(tauriConfig.productName).toBe('WordCovenant')
    expect(tauriConfig.identifier).toBe('com.wordcovenant.desktop')
    expect(JSON.stringify(capability)).not.toContain('shell:')
    expect(JSON.stringify(tauriConfig)).not.toContain('http://*')
    expect(JSON.stringify(tauriConfig)).not.toContain('https://*')
  })
})
