import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { createHash } from 'node:crypto'
import { mkdtemp, mkdir, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { afterEach, describe, test } from 'node:test'
import { loadBundledModelMetadata, stageBundledModel, verifyBundledModel } from './stage-bundled-model.mjs'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'

const temporaryDirectories = []
const PROJECT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const execFileAsync = promisify(execFile)

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map(directory => rm(directory, { force: true, recursive: true })))
})

async function createFixture(bytes, manifestOverrides = {}) {
  const directory = await mkdtemp(join(tmpdir(), 'word-covenant-bundled-model-'))
  temporaryDirectories.push(directory)

  const sourcePath = join(directory, 'source', 'ggml-base.bin')
  const lockPath = join(directory, 'models', 'whisper-base.lock.json')
  const manifestPath = join(directory, 'resources', 'models', 'manifest.json')
  const destinationPath = join(directory, 'resources', 'models', 'ggml-base.bin')
  const sha256 = createHash('sha256').update(bytes).digest('hex')
  const metadata = {
    schemaVersion: 1,
    modelId: '32ce7670-d303-4566-9cc3-123a380befe9',
    modelKind: 'speech_recognition',
    inputFormat: 'whisper.cpp-ggml',
    variant: 'base',
    multilingual: true,
    artifactFileName: 'ggml-base.bin',
    sizeBytes: bytes.byteLength,
    sha256,
    version: 'fixture-v1',
    modelCardId: 'word-covenant/test-model',
    licenseId: 'MIT',
    licenseConfirmedAt: '2026-08-10T00:00:00Z',
    source: {
      repository: 'word-covenant/test-models',
      revision: '0123456789abcdef0123456789abcdef01234567',
      url: 'https://example.invalid/0123456789abcdef0123456789abcdef01234567/ggml-base.bin',
    },
  }

  await Promise.all([
    mkdir(dirname(sourcePath), { recursive: true }),
    mkdir(dirname(lockPath), { recursive: true }),
    mkdir(dirname(manifestPath), { recursive: true }),
  ])
  await Promise.all([
    writeFile(sourcePath, bytes),
    writeFile(lockPath, `${JSON.stringify(metadata, null, 2)}\n`),
    writeFile(manifestPath, `${JSON.stringify({ ...metadata, ...manifestOverrides }, null, 2)}\n`),
  ])

  return { destinationPath, lockPath, manifestPath, sourcePath }
}

describe('stageBundledModel', () => {
  test('keeps the committed lock, bundled manifest, license notice, and Tauri resource mapping aligned', async () => {
    const lockPath = join(PROJECT_ROOT, 'models', 'whisper-base.lock.json')
    const manifestPath = join(PROJECT_ROOT, 'src-tauri', 'resources', 'models', 'manifest.json')
    const tauriConfigPath = join(PROJECT_ROOT, 'src-tauri', 'tauri.conf.json')
    const licensePath = join(PROJECT_ROOT, 'src-tauri', 'resources', 'third-party', 'whisper-base-model-MIT.txt')
    const modelCardPath = join(PROJECT_ROOT, 'src-tauri', 'resources', 'third-party', 'whisper-base-model-card.txt')
    const [metadata, tauriConfigText, licenseText, modelCardText] = await Promise.all([
      loadBundledModelMetadata({ lockPath, manifestPath }),
      readFile(tauriConfigPath, 'utf8'),
      readFile(licensePath, 'utf8'),
      readFile(modelCardPath, 'utf8'),
    ])
    const tauriConfig = JSON.parse(tauriConfigText)

    assert.equal(metadata.modelId, '32ce7670-d303-4566-9cc3-123a380befe9')
    assert.equal(metadata.modelKind, 'speech_recognition')
    assert.equal(metadata.inputFormat, 'whisper.cpp-ggml')
    assert.equal(metadata.variant, 'base')
    assert.equal(metadata.multilingual, true)
    assert.equal(metadata.sha256, '60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe')
    assert.equal(metadata.sizeBytes, 147951465)
    assert.equal(metadata.modelCardId, 'openai/whisper-base')
    assert.equal(metadata.licenseId, 'MIT')
    assert.equal(metadata.source.repository, 'ggerganov/whisper.cpp')
    assert.equal(metadata.source.revision, '5359861c739e955e79d9a303bcbc70fb988958b1')
    assert.equal(
      metadata.source.url,
      'https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-base.bin'
    )
    assert.equal(tauriConfig.bundle.targets, 'dmg')
    assert.deepEqual(tauriConfig.bundle.resources, {
      'resources/models/ggml-base.bin': 'models/ggml-base.bin',
      'resources/models/manifest.json': 'models/manifest.json',
      'resources/third-party/whisper-base-model-card.txt': 'third-party/whisper-base-model-card.txt',
      'resources/third-party/whisper-base-model-MIT.txt': 'third-party/whisper-base-model-MIT.txt',
    })
    assert.match(licenseText, /MIT License/)
    assert.match(modelCardText, /Model card ID: openai\/whisper-base/)
  })

  test('copies only a source whose byte count and SHA-256 match the mirrored manifest', async () => {
    const bytes = Buffer.from('verified local model fixture')
    const fixture = await createFixture(bytes)

    const result = await stageBundledModel(fixture)

    assert.equal(result.destinationPath, fixture.destinationPath)
    assert.equal(result.sha256, createHash('sha256').update(bytes).digest('hex'))
    assert.equal(result.sizeBytes, bytes.byteLength)
    assert.deepEqual(await readFile(fixture.destinationPath), bytes)
    assert.equal((await stat(fixture.destinationPath)).size, bytes.byteLength)
    if (process.platform !== 'win32') {
      assert.equal((await stat(fixture.destinationPath)).mode & 0o777, 0o644)
    }
  })

  test('rejects a resource manifest that diverges from the reviewed lock', async () => {
    const fixture = await createFixture(Buffer.from('verified local model fixture'), {
      version: 'unreviewed-v2',
    })

    await assert.rejects(
      () => stageBundledModel(fixture),
      /resource manifest must exactly match the reviewed model lock/
    )
    await assert.rejects(() => stat(fixture.destinationPath))
  })

  test('verifies an existing staged artifact without staging a second copy', async () => {
    const bytes = Buffer.from('verified local model fixture')
    const fixture = await createFixture(bytes)
    await stageBundledModel(fixture)

    const result = await verifyBundledModel({
      lockPath: fixture.lockPath,
      manifestPath: fixture.manifestPath,
      artifactPath: fixture.destinationPath,
    })

    assert.equal(result.artifactPath, fixture.destinationPath)
    assert.equal(result.sizeBytes, bytes.byteLength)
    assert.deepEqual(await readFile(fixture.destinationPath), bytes)
  })

  test('CLI verification always checks the staged destination instead of an overlay source', async () => {
    const bytes = Buffer.from('verified local model fixture')
    const fixture = await createFixture(bytes)
    const scriptPath = join(PROJECT_ROOT, 'scripts', 'stage-bundled-model.mjs')
    await stageBundledModel(fixture)

    await execFileAsync(
      process.execPath,
      [
        scriptPath,
        '--verify',
        '--lock',
        fixture.lockPath,
        '--manifest',
        fixture.manifestPath,
        '--destination',
        fixture.destinationPath,
      ],
      {
        env: {
          ...process.env,
          WORD_COVENANT_MODEL_OVERLAY: join(dirname(fixture.sourcePath), 'missing-overlay.bin'),
        },
      }
    )
  })

  test('does not stage a source whose SHA-256 differs from the reviewed lock', async () => {
    const fixture = await createFixture(Buffer.from('verified local model fixture'))
    await writeFile(fixture.sourcePath, Buffer.from('tampered local model fixture'))

    await assert.rejects(() => stageBundledModel(fixture), /SHA-256 did not match/)
    await assert.rejects(() => stat(fixture.destinationPath))
  })

  test('accepts pnpm-style argument separation for explicit local sources', async () => {
    const bytes = Buffer.from('verified local model fixture')
    const fixture = await createFixture(bytes)
    const scriptPath = join(PROJECT_ROOT, 'scripts', 'stage-bundled-model.mjs')

    await execFileAsync(process.execPath, [
      scriptPath,
      '--',
      '--source',
      fixture.sourcePath,
      '--lock',
      fixture.lockPath,
      '--manifest',
      fixture.manifestPath,
      '--destination',
      fixture.destinationPath,
    ])

    assert.deepEqual(await readFile(fixture.destinationPath), bytes)
  })
})
