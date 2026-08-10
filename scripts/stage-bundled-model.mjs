import { createHash, randomUUID } from 'node:crypto'
import { chmod, open, lstat, mkdir, readFile, rename, rm } from 'node:fs/promises'
import { basename, dirname, isAbsolute, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { isDeepStrictEqual } from 'node:util'

const SCRIPT_DIRECTORY = dirname(fileURLToPath(import.meta.url))
const PROJECT_ROOT = resolve(SCRIPT_DIRECTORY, '..')
const SHA256_PATTERN = /^[a-f0-9]{64}$/
const UUID_PATTERN = /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/
const REVISION_PATTERN = /^[a-f0-9]{40}$/
const COPY_BUFFER_BYTES = 64 * 1024

function fail(message) {
  throw new Error(message)
}

function requireString(value, field) {
  if (typeof value !== 'string' || value.length === 0) {
    fail(`${field} must be a non-empty string`)
  }
  return value
}

function assertModelMetadata(metadata, label) {
  if (metadata === null || typeof metadata !== 'object' || Array.isArray(metadata)) {
    fail(`${label} must be a JSON object`)
  }
  if (metadata.schemaVersion !== 1) {
    fail(`${label}.schemaVersion must be 1`)
  }
  if (!UUID_PATTERN.test(requireString(metadata.modelId, `${label}.modelId`))) {
    fail(`${label}.modelId must be a UUID`)
  }
  if (metadata.modelKind !== 'speech_recognition') {
    fail(`${label}.modelKind must be speech_recognition`)
  }
  if (metadata.inputFormat !== 'whisper.cpp-ggml') {
    fail(`${label}.inputFormat must be whisper.cpp-ggml`)
  }
  if (metadata.variant !== 'base') {
    fail(`${label}.variant must be base`)
  }
  if (metadata.multilingual !== true) {
    fail(`${label}.multilingual must be true`)
  }

  const artifactFileName = requireString(metadata.artifactFileName, `${label}.artifactFileName`)
  if (
    artifactFileName !== basename(artifactFileName) ||
    artifactFileName === '.' ||
    artifactFileName === '..' ||
    isAbsolute(artifactFileName)
  ) {
    fail(`${label}.artifactFileName must be a single file name`)
  }
  if (!Number.isSafeInteger(metadata.sizeBytes) || metadata.sizeBytes <= 0) {
    fail(`${label}.sizeBytes must be a positive safe integer`)
  }
  if (!SHA256_PATTERN.test(requireString(metadata.sha256, `${label}.sha256`))) {
    fail(`${label}.sha256 must be a lowercase SHA-256 digest`)
  }
  requireString(metadata.version, `${label}.version`)
  requireString(metadata.modelCardId, `${label}.modelCardId`)
  requireString(metadata.licenseId, `${label}.licenseId`)
  const confirmedAt = requireString(metadata.licenseConfirmedAt, `${label}.licenseConfirmedAt`)
  if (Number.isNaN(Date.parse(confirmedAt))) {
    fail(`${label}.licenseConfirmedAt must be an ISO-8601 timestamp`)
  }

  const source = metadata.source
  if (source === null || typeof source !== 'object' || Array.isArray(source)) {
    fail(`${label}.source must be a JSON object`)
  }
  requireString(source.repository, `${label}.source.repository`)
  if (!REVISION_PATTERN.test(requireString(source.revision, `${label}.source.revision`))) {
    fail(`${label}.source.revision must be a 40-character lowercase revision`)
  }
  const sourceUrl = requireString(source.url, `${label}.source.url`)
  let parsedSourceUrl
  try {
    parsedSourceUrl = new URL(sourceUrl)
  } catch {
    fail(`${label}.source.url must be an HTTPS URL`)
  }
  if (parsedSourceUrl.protocol !== 'https:') {
    fail(`${label}.source.url must be an HTTPS URL`)
  }
  if (!sourceUrl.includes(source.revision)) {
    fail(`${label}.source.url must pin the declared source revision`)
  }
}

async function readJson(path, label) {
  let text
  try {
    text = await readFile(path, 'utf8')
  } catch (error) {
    fail(`could not read ${label}: ${error.message}`)
  }
  try {
    return JSON.parse(text)
  } catch (error) {
    fail(`could not parse ${label}: ${error.message}`)
  }
}

async function loadReviewedMetadata(lockPath, manifestPath) {
  const [lock, manifest] = await Promise.all([
    readJson(lockPath, 'reviewed model lock'),
    readJson(manifestPath, 'bundled resource manifest'),
  ])
  assertModelMetadata(lock, 'reviewed model lock')
  assertModelMetadata(manifest, 'bundled resource manifest')
  if (!isDeepStrictEqual(lock, manifest)) {
    fail('resource manifest must exactly match the reviewed model lock')
  }
  return lock
}

export async function loadBundledModelMetadata({ lockPath, manifestPath }) {
  return loadReviewedMetadata(resolve(lockPath), resolve(manifestPath))
}

async function readAndVerifyModelSource(sourcePath, metadata, onChunk) {
  const sourceLinkMetadata = await lstat(sourcePath).catch(error => {
    fail(`could not inspect model source: ${error.message}`)
  })
  if (!sourceLinkMetadata.isFile()) {
    fail('model source must be a regular file')
  }

  let source
  try {
    source = await open(sourcePath, 'r')
    const sourceMetadata = await source.stat()
    if (!sourceMetadata.isFile()) {
      fail('model source must be a regular file')
    }
    const hash = createHash('sha256')
    const buffer = Buffer.allocUnsafe(COPY_BUFFER_BYTES)
    let sizeBytes = 0
    let position = 0
    while (true) {
      const { bytesRead } = await source.read(buffer, 0, buffer.length, position)
      if (bytesRead === 0) {
        break
      }
      const chunk = buffer.subarray(0, bytesRead)
      hash.update(chunk)
      await onChunk?.(chunk)
      sizeBytes += bytesRead
      position += bytesRead
    }
    const sha256 = hash.digest('hex')
    if (sizeBytes !== metadata.sizeBytes) {
      fail(`model source size did not match the reviewed lock: expected ${metadata.sizeBytes}, got ${sizeBytes}`)
    }
    if (sha256 !== metadata.sha256) {
      fail('model source SHA-256 did not match the reviewed lock')
    }
    return { sha256, sizeBytes }
  } finally {
    await source?.close()
  }
}

async function copyAndVerify(sourcePath, temporaryPath, metadata) {
  let temporary
  try {
    temporary = await open(temporaryPath, 'wx', 0o600)
    const verification = await readAndVerifyModelSource(sourcePath, metadata, chunk => writeAll(temporary, chunk))
    await temporary.sync()
    return verification
  } finally {
    await temporary?.close()
  }
}

async function writeAll(file, buffer) {
  let offset = 0
  while (offset < buffer.byteLength) {
    const { bytesWritten } = await file.write(buffer, offset, buffer.byteLength - offset, null)
    if (bytesWritten === 0) {
      fail('could not write the staged model artifact')
    }
    offset += bytesWritten
  }
}

export async function verifyBundledModel({ lockPath, manifestPath, artifactPath }) {
  if (!artifactPath) {
    fail('provide a local bundled model artifact to verify')
  }
  const [resolvedLockPath, resolvedManifestPath, resolvedArtifactPath] = [
    resolve(lockPath),
    resolve(manifestPath),
    resolve(artifactPath),
  ]
  const metadata = await loadReviewedMetadata(resolvedLockPath, resolvedManifestPath)
  if (basename(resolvedArtifactPath) !== metadata.artifactFileName) {
    fail('bundled resource file name must match the reviewed artifactFileName')
  }
  const verification = await readAndVerifyModelSource(resolvedArtifactPath, metadata)
  return { artifactPath: resolvedArtifactPath, ...verification }
}

export async function stageBundledModel({ lockPath, manifestPath, sourcePath, destinationPath }) {
  if (!sourcePath) {
    fail('provide a local model source with --source or WORD_COVENANT_MODEL_OVERLAY')
  }
  const [resolvedLockPath, resolvedManifestPath, resolvedSourcePath, resolvedDestinationPath] = [
    resolve(lockPath),
    resolve(manifestPath),
    resolve(sourcePath),
    resolve(destinationPath),
  ]
  const metadata = await loadReviewedMetadata(resolvedLockPath, resolvedManifestPath)
  if (basename(resolvedDestinationPath) !== metadata.artifactFileName) {
    fail('staged resource file name must match the reviewed artifactFileName')
  }

  if (resolvedSourcePath === resolvedDestinationPath) {
    const verification = await readAndVerifyModelSource(resolvedSourcePath, metadata)
    return { destinationPath: resolvedDestinationPath, ...verification }
  }

  const destinationDirectory = dirname(resolvedDestinationPath)
  await mkdir(destinationDirectory, { recursive: true })
  const temporaryPath = resolve(
    destinationDirectory,
    `.${metadata.artifactFileName}.stage-${process.pid}-${randomUUID()}`
  )
  try {
    const verification = await copyAndVerify(resolvedSourcePath, temporaryPath, metadata)
    await chmod(temporaryPath, 0o644)
    await rename(temporaryPath, resolvedDestinationPath)
    await chmod(resolvedDestinationPath, 0o644)
    return { destinationPath: resolvedDestinationPath, ...verification }
  } finally {
    await rm(temporaryPath, { force: true })
  }
}

function parseArguments(argv) {
  const options = {
    lockPath: resolve(PROJECT_ROOT, 'models/whisper-base.lock.json'),
    manifestPath: resolve(PROJECT_ROOT, 'src-tauri/resources/models/manifest.json'),
    sourcePath: process.env.WORD_COVENANT_MODEL_OVERLAY,
    destinationPath: resolve(PROJECT_ROOT, 'src-tauri/resources/models/ggml-base.bin'),
    verify: false,
  }
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index]
    if (flag === '--') {
      continue
    }
    if (flag === '--help') {
      return null
    }
    if (flag === '--verify') {
      options.verify = true
      continue
    }
    const value = argv[index + 1]
    if (!value || value.startsWith('--')) {
      fail(`${flag} requires a value`)
    }
    if (flag === '--source') {
      options.sourcePath = value
    } else if (flag === '--lock') {
      options.lockPath = value
    } else if (flag === '--manifest') {
      options.manifestPath = value
    } else if (flag === '--destination') {
      options.destinationPath = value
    } else {
      fail(`unknown argument: ${flag}`)
    }
    index += 1
  }
  return options
}

function printUsage() {
  console.log('Usage: node scripts/stage-bundled-model.mjs --source <local-model-file>')
  console.log('       WORD_COVENANT_MODEL_OVERLAY=<local-model-file> node scripts/stage-bundled-model.mjs')
  console.log('       node scripts/stage-bundled-model.mjs --verify [--source <staged-model-file>]')
}

async function main() {
  const options = parseArguments(process.argv.slice(2))
  if (options === null) {
    printUsage()
    return
  }
  if (options.verify) {
    const result = await verifyBundledModel({
      lockPath: options.lockPath,
      manifestPath: options.manifestPath,
      artifactPath: options.destinationPath,
    })
    console.log(`Verified bundled model: ${result.artifactPath}`)
    return
  }
  const result = await stageBundledModel(options)
  console.log(`Staged verified bundled model: ${result.destinationPath}`)
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => {
    console.error(`Could not stage bundled model: ${error.message}`)
    process.exitCode = 1
  })
}
