# Audit Keychain Seal Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让 macOS 本地审计链能够检测 SQLite 数据库被离线重写或截断，同时在写入与 Keychain 更新之间发生崩溃时可靠恢复。

**Architecture:** 保留 SQLite 中可审计的 SHA-256 事件链，但将当前链尾的可信锚点保存在 macOS Keychain。每次事件写入先在 Keychain 建立带前序哈希的 pending 锚点，再提交 SQLite 事务，最后将 pending 原子地提升为当前锚点。重启时只接受三种状态：已锚定的当前尾部、已提交但尚未提升的 pending 尾部，或尚未提交且仍为旧尾部；任何其他状态都视为完整性失败。

**Tech Stack:** Rust、rusqlite、`security-framework`（macOS Keychain）、SHA-256、Tauri 2、Rust unit tests。

---

## 威胁边界

当前无密钥 SHA-256 链只能发现没有同步重算链的直接字段修改。能够读写 SQLite 的攻击者可以删除尾部事件，或重建 `previous_hash` 和 `hash`，因此它不是抗重写证据。

本计划抵抗的对象是“可读写应用 SQLite 与模型文件，但不能修改当前 macOS 登录钥匙串条目”的本地攻击者。它不能抵抗已经获得用户 Keychain 授权、删除整个应用数据和 Keychain 条目、或回滚整个用户帐户快照的攻击；这些限制必须在导出/恢复文档中明确说明。旧数据库在首次升级前的历史不能被追溯证明，只能被明确标记为已迁移。

### Task 1: 定义可测试的审计封印协议

**Files:**
- Create: `src-tauri/src/audit/seal.rs`
- Modify: `src-tauri/src/audit/mod.rs`
- Test: `src-tauri/src/audit/seal.rs`

**Step 1: 写失败测试。**

定义不依赖 Keychain 的 `AuditSealStore` trait 和 `AuditSeal` 值：`anchored_hash`、`pending_previous_hash`、`pending_hash`。用内存实现验证以下恢复规则：

```rust
assert_eq!(seal.reconcile(Some(old_hash)), Reconcile::ClearPending);
assert_eq!(seal.reconcile(Some(new_hash)), Reconcile::FinalizePending);
assert!(seal.reconcile(Some(unrelated_hash)).is_err());
```

**Step 2: 运行失败测试。**

Run: `cargo test --manifest-path src-tauri/Cargo.toml audit::seal --lib`

Expected: FAIL，因为 trait、seal 类型和恢复规则尚不存在。

**Step 3: 实现最小状态机。**

`prepare(previous_hash, next_hash)` 只在当前锚点与 `previous_hash` 一致且无 pending 时写入 pending。`finalize(next_hash)` 只接受已匹配的 pending。`reconcile(database_tail)` 只允许上述三种崩溃状态，其他分叉或截断返回 `AuditSealError::Integrity`。

**Step 4: 运行测试。**

Run: `cargo test --manifest-path src-tauri/Cargo.toml audit::seal --lib`

Expected: PASS。

**Step 5: Commit。**

```bash
git add src-tauri/src/audit/seal.rs src-tauri/src/audit/mod.rs
git commit -m "feat: define recoverable audit seal protocol"
```

### Task 2: 实现 macOS Keychain 封印存储

**Files:**
- Modify: `src-tauri/Cargo.toml`
- Modify: `src-tauri/Cargo.lock`
- Create: `src-tauri/src/audit/keychain_seal.rs`
- Test: `src-tauri/src/audit/keychain_seal.rs`

**Step 1: 写账户定位测试。**

使用数据库的规范化绝对路径的 SHA-256 作为 Keychain account；service 固定为 `com.wordcovenant.audit-seal.v1`。测试不同路径不冲突，等价路径映射到同一 account，路径文本本身不写入 Keychain value。

**Step 2: 实现 Keychain 后端。**

在 macOS target dependencies 加入已审计的 `security-framework`，封装 `get_generic_password` 与 `set_generic_password`。value 为版本化 JSON，仅存上述 hash 和 pending 信息，不存转写、模型路径或音频。将 `errSecItemNotFound` 映射为首次启动的 `None`；其他 Keychain 错误必须失败关闭。

**Step 3: 防止错误的首次初始化。**

当 Keychain 没有锚点且 SQLite 为空时才建立空锚点。若 SQLite 已有事件但没有 Keychain 条目，返回 `MissingAuditSeal`，除非调用显式的“迁移旧数据库”路径；不要静默信任已有数据库。

**Step 4: 运行测试。**

Run: `cargo test --manifest-path src-tauri/Cargo.toml audit::keychain_seal --lib`

Expected: PASS。Keychain 实际集成仅用手工 macOS smoke test 验证，普通单元测试使用内存后端。

**Step 5: Commit。**

```bash
git add src-tauri/Cargo.toml src-tauri/Cargo.lock src-tauri/src/audit/keychain_seal.rs
git commit -m "feat: add macOS keychain audit seal storage"
```

### Task 3: 将所有审计写入接入两阶段封印

**Files:**
- Modify: `src-tauri/src/state.rs`
- Modify: `src-tauri/src/audit/store.rs`
- Modify: `src-tauri/src/audit/hash_chain.rs`
- Test: `src-tauri/src/state.rs`

**Step 1: 写失败测试。**

用内存封印后端覆盖：正常 append、SQLite 提交前失败、SQLite 提交后/Keychain finalize 前的模拟崩溃、尾部删除、全链重写和错误 pending。每种失败后重新构建 `AppState`，只有正常写入和可解释的两种 crash 状态能打开。

**Step 2: 引入协调器。**

把 `record_audit`、capture gap/segment、最终 transcript revision、模型导入的事件写入收束到一个协调器：

1. 从内存 trail 取得 `previous_hash` 和待写 event 的 `hash`。
2. `seal.prepare(previous_hash, event.hash)`。
3. 执行现有 SQLite 单事务写入。
4. 更新内存 trail。
5. `seal.finalize(event.hash)`。

若步骤 3 失败，清除 matching pending；若进程在步骤 3/5 之间终止，下一次 `reconcile` 决定是否安全提升。步骤 5 失败不可把 event 当作未写入；应在同一进程标记 seal pending 并拒绝下一次写入，直到可安全 finalize。

**Step 3: 保持数据库验证职责清晰。**

`AuditStore::verify()` 继续验证 SQLite 内链和绑定记录；`AppState::open()` 再验证 Keychain seal。不要把 Keychain secret、anchor JSON 或绝对模型路径暴露给 WebView/审计 payload。

**Step 4: 运行测试。**

Run: `cargo test --manifest-path src-tauri/Cargo.toml state::tests --lib`

Expected: PASS，包含重写/截断会失败打开的回归。

**Step 5: Commit。**

```bash
git add src-tauri/src/state.rs src-tauri/src/audit/store.rs src-tauri/src/audit/hash_chain.rs
git commit -m "feat: seal audit writes with keychain anchor"
```

### Task 4: 迁移、错误呈现与文档

**Files:**
- Modify: `src-tauri/src/state.rs`
- Modify: `src-tauri/src/commands.rs`
- Modify: `src/types.ts`
- Modify: `src/components/PrivacyStatus.vue`
- Modify: `README.md`
- Modify: `docs/plans/2026-08-07-word-covenant-roadmap.md`
- Test: `src-tauri/src/state.rs`

**Step 1: 设计显式旧数据库迁移。**

如果产品需要保留已有开发期 SQLite，提供只在本地 UI 明示的迁移操作：显示“升级前历史不能反向验证”，读取当前尾部、写入 Keychain anchor、追加 `AuditSealMigrated` 审计事件。默认启动不自动迁移。

**Step 2: 显示可操作但不泄露细节的状态。**

用户只看到“本地完整性验证通过 / 需要恢复或迁移”；不要显示 Keychain account、哈希、数据库路径或原始错误。应用无法验证时必须拒绝启动敏感写入，而非静默建立新链。

**Step 3: 修正文档声明。**

README 和路线图需要区分：当前链的绑定能力、Keychain anchor 的威胁模型、以及它不是法律意义的不可否认性。不得声称可以抵抗具有 Keychain 权限的攻击者。

**Step 4: 运行测试。**

Run: `pnpm test --run && pnpm type-check`

Expected: PASS。

**Step 5: Commit。**

```bash
git add src-tauri/src/state.rs src-tauri/src/commands.rs src/types.ts src/components/PrivacyStatus.vue README.md docs/plans/2026-08-07-word-covenant-roadmap.md
git commit -m "docs: explain keychain audit seal guarantees"
```

### Task 5: 最终验证与 macOS 手工恢复演练

**Files:**
- Modify: `docs/plans/2026-08-08-m1-macos-real-capture-manual-acceptance.md`

**Step 1: 跑完整本地验证。**

Run:

```bash
cargo fmt --manifest-path src-tauri/Cargo.toml --check
cargo test --manifest-path src-tauri/Cargo.toml --offline
cargo check --manifest-path src-tauri/Cargo.toml --release --offline
pnpm test --run
pnpm type-check
pnpm build
git diff --check
```

Expected: 全部通过，且不需要新增网络访问。

**Step 2: 手工 macOS 验收。**

在一个新本地数据库里写入 session、模型与最终转写；退出应用；用 SQLite 直接删除尾部或重写 chain；重开应用应报告完整性失败。分别模拟 SQLite 提交前和 Keychain finalize 前退出，重开应能按 pending 规则恢复。记录设备、macOS 版本、是否出现 Keychain 授权提示和实际结果。

**Step 3: Commit。**

```bash
git add docs/plans/2026-08-08-m1-macos-real-capture-manual-acceptance.md
git commit -m "test: document audit seal recovery acceptance"
```
