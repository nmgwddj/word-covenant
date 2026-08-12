#[cfg(target_os = "macos")]
mod macos {
    use hound::{SampleFormat, WavReader};
    use serde::{Deserialize, Serialize};
    use std::collections::{BTreeMap, BTreeSet};
    use std::env;
    use std::fs;
    use std::path::{Component, Path, PathBuf};
    use std::time::{Duration, Instant};
    use uuid::Uuid;
    use word_covenant_lib::diarization::{
        bundled_speaker_model, match_speaker_profile, AnonymousSpeakerAssignment,
        OnnxSpeakerEmbeddingEngine, SessionSpeakerClusterer, SpeakerEmbedding,
        SpeakerEmbeddingEngine, SpeakerMatchCandidate, SpeakerMatchDecision, SpeakerMatchPolicy,
        SpeakerSampleQuality,
    };
    use word_covenant_lib::inference::{
        InferenceAudioWindow, INFERENCE_CHANNELS, INFERENCE_SAMPLE_RATE_HZ,
        MAX_INFERENCE_WINDOW_SAMPLES,
    };

    const FIXTURE_DIRECTORY_ENV: &str = "WORD_COVENANT_SPEAKER_FIXTURES";
    const RESOURCE_DIRECTORY_ENV: &str = "WORD_COVENANT_SPEAKER_RESOURCES";

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase", deny_unknown_fields)]
    struct FixtureManifest {
        schema_version: u32,
        corpus_id: String,
        consent: FixtureConsent,
        fixtures: Vec<FixtureEntry>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase", deny_unknown_fields)]
    struct FixtureConsent {
        confirmed: bool,
        statement: String,
        collected_at: String,
    }

    #[derive(Clone, Debug, Deserialize)]
    #[serde(rename_all = "camelCase", deny_unknown_fields)]
    struct FixtureEntry {
        id: String,
        speaker_id: String,
        session_id: String,
        file: PathBuf,
        language: FixtureLanguage,
        condition: FixtureCondition,
        split: FixtureSplit,
        role: FixtureRole,
    }

    #[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
    #[serde(rename_all = "snake_case")]
    enum FixtureLanguage {
        Chinese,
        English,
        Mixed,
    }

    #[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
    #[serde(rename_all = "snake_case")]
    enum FixtureCondition {
        Near,
        Distance,
        Noise,
        Overlap,
    }

    #[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
    #[serde(rename_all = "snake_case")]
    enum FixtureSplit {
        Calibration,
        Acceptance,
    }

    #[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
    #[serde(rename_all = "snake_case")]
    enum FixtureRole {
        Enrollment,
        Probe,
    }

    struct EmbeddedFixture {
        metadata: FixtureEntry,
        embedding: SpeakerEmbedding,
        quality: SpeakerSampleQuality,
        audio_duration: Duration,
        inference_duration: Duration,
    }

    #[derive(Debug, Serialize)]
    #[serde(rename_all = "camelCase")]
    struct BenchmarkReport {
        schema_version: u32,
        corpus_id: String,
        model_id: String,
        model_version: String,
        fixture_count: usize,
        speaker_count: usize,
        session_count: usize,
        thresholds_are_provisional: bool,
        calibration: SplitReport,
        acceptance: SplitReport,
        dimension_slices: Vec<DimensionSliceReport>,
        inference: InferenceReport,
        privacy: PrivacyReport,
    }

    #[derive(Debug, Serialize)]
    #[serde(rename_all = "camelCase")]
    struct DimensionSliceReport {
        split: FixtureSplit,
        dimension: &'static str,
        value: &'static str,
        metrics: SplitReport,
    }

    #[derive(Debug, Default, Serialize)]
    #[serde(rename_all = "camelCase")]
    struct SplitReport {
        fixture_count: usize,
        verification_same_speaker_trials: usize,
        verification_different_speaker_trials: usize,
        false_reject_count: usize,
        false_accept_count: usize,
        false_reject_rate: Option<f64>,
        false_accept_rate: Option<f64>,
        profile_probe_count: usize,
        correct_profile_count: usize,
        incorrect_profile_count: usize,
        uncertainty_count: usize,
        uncertainty_rate: Option<f64>,
        anonymous_pair_trials: usize,
        anonymous_pair_confusion_count: usize,
        anonymous_pair_confusion_rate: Option<f64>,
    }

    #[derive(Debug, Serialize)]
    #[serde(rename_all = "camelCase")]
    struct InferenceReport {
        model_load_ms: f64,
        embedding_p50_ms: f64,
        embedding_p95_ms: f64,
        embedding_max_ms: f64,
        realtime_factor_p95: f64,
        peak_resident_memory_bytes: Option<u64>,
        power_measurement: &'static str,
    }

    #[derive(Debug, Serialize)]
    #[serde(rename_all = "camelCase")]
    struct PrivacyReport {
        fixture_audio_persisted: bool,
        embeddings_persisted: bool,
        network_client_used: bool,
        external_egress_observation: &'static str,
    }

    pub fn main() -> Result<(), String> {
        let fixture_root = match env::var_os(FIXTURE_DIRECTORY_ENV) {
            Some(path) => canonical_directory(Path::new(&path))?,
            None => {
                println!(
                    "speaker recognition benchmark not run: set {FIXTURE_DIRECTORY_ENV} to a consented local fixture directory"
                );
                return Ok(());
            }
        };
        let manifest = load_fixture_manifest(&fixture_root)?;
        validate_fixture_manifest(&manifest)?;

        let resource_root = match env::var_os(RESOURCE_DIRECTORY_ENV) {
            Some(path) => canonical_directory(Path::new(&path))?,
            None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources"),
        };
        let bundled = bundled_speaker_model(&resource_root)?;
        let profile_policy = bundled.manifest.profile_match_policy()?;
        let anonymous_policy = bundled.manifest.anonymous_match_policy()?;
        let model_id = bundled.manifest.provenance()?.model_id().to_owned();
        let model_version = bundled.manifest.provenance()?.model_version().to_owned();
        let model_load_started = Instant::now();
        let mut engine = OnnxSpeakerEmbeddingEngine::from_bundled(bundled)?;
        let model_load_duration = model_load_started.elapsed();

        let mut embedded = Vec::with_capacity(manifest.fixtures.len());
        for fixture in manifest.fixtures.iter().cloned() {
            let samples = read_fixture_wav(&fixture_root, &fixture.file)?;
            let audio_duration =
                Duration::from_secs_f64(samples.len() as f64 / f64::from(INFERENCE_SAMPLE_RATE_HZ));
            let capture_end_ns = u64::try_from(audio_duration.as_nanos())
                .map_err(|_| format!("fixture {} duration overflows", fixture.id))?;
            let window = InferenceAudioWindow::new(
                Uuid::new_v4(),
                0,
                capture_end_ns,
                INFERENCE_SAMPLE_RATE_HZ,
                INFERENCE_CHANNELS,
                samples,
            )?;
            let started = Instant::now();
            let (embedding, quality) = engine
                .embed(&window)
                .map_err(|error| format!("fixture {} embedding failed: {error}", fixture.id))?;
            embedded.push(EmbeddedFixture {
                metadata: fixture,
                embedding,
                quality,
                audio_duration,
                inference_duration: started.elapsed(),
            });
        }

        let report = BenchmarkReport {
            schema_version: 1,
            corpus_id: manifest.corpus_id,
            model_id,
            model_version,
            fixture_count: embedded.len(),
            speaker_count: unique_count(&embedded, |fixture| &fixture.metadata.speaker_id),
            session_count: unique_count(&embedded, |fixture| &fixture.metadata.session_id),
            thresholds_are_provisional: true,
            calibration: evaluate_split(
                &embedded,
                FixtureSplit::Calibration,
                profile_policy,
                anonymous_policy,
            ),
            acceptance: evaluate_split(
                &embedded,
                FixtureSplit::Acceptance,
                profile_policy,
                anonymous_policy,
            ),
            dimension_slices: dimension_slices(&embedded, profile_policy, anonymous_policy),
            inference: inference_report(&embedded, model_load_duration),
            privacy: PrivacyReport {
                fixture_audio_persisted: false,
                embeddings_persisted: false,
                network_client_used: false,
                external_egress_observation:
                    "required: observe the benchmark PID with the host network monitor",
            },
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&report)
                .map_err(|error| format!("benchmark report serialization failed: {error}"))?
        );
        Ok(())
    }

    fn evaluate_split(
        fixtures: &[EmbeddedFixture],
        split: FixtureSplit,
        profile_policy: SpeakerMatchPolicy,
        anonymous_policy: SpeakerMatchPolicy,
    ) -> SplitReport {
        let selected = fixtures
            .iter()
            .filter(|fixture| fixture.metadata.split == split)
            .collect::<Vec<_>>();
        evaluate_selected(&selected, profile_policy, anonymous_policy)
    }

    fn evaluate_selected(
        selected: &[&EmbeddedFixture],
        profile_policy: SpeakerMatchPolicy,
        anonymous_policy: SpeakerMatchPolicy,
    ) -> SplitReport {
        let mut report = SplitReport {
            fixture_count: selected.len(),
            ..SplitReport::default()
        };
        evaluate_verification(selected, profile_policy, &mut report);
        evaluate_profiles(selected, profile_policy, &mut report);
        evaluate_clustering(selected, anonymous_policy, &mut report);
        report.false_reject_rate = rate(
            report.false_reject_count,
            report.verification_same_speaker_trials,
        );
        report.false_accept_rate = rate(
            report.false_accept_count,
            report.verification_different_speaker_trials,
        );
        report.uncertainty_rate = rate(report.uncertainty_count, report.profile_probe_count);
        report.anonymous_pair_confusion_rate = rate(
            report.anonymous_pair_confusion_count,
            report.anonymous_pair_trials,
        );
        report
    }

    fn dimension_slices(
        fixtures: &[EmbeddedFixture],
        profile_policy: SpeakerMatchPolicy,
        anonymous_policy: SpeakerMatchPolicy,
    ) -> Vec<DimensionSliceReport> {
        let mut reports = Vec::new();
        for split in [FixtureSplit::Calibration, FixtureSplit::Acceptance] {
            for language in [
                FixtureLanguage::Chinese,
                FixtureLanguage::English,
                FixtureLanguage::Mixed,
            ] {
                let selected = fixtures
                    .iter()
                    .filter(|fixture| {
                        fixture.metadata.split == split && fixture.metadata.language == language
                    })
                    .collect::<Vec<_>>();
                reports.push(DimensionSliceReport {
                    split,
                    dimension: "language",
                    value: language.as_str(),
                    metrics: evaluate_selected(&selected, profile_policy, anonymous_policy),
                });
            }
            for condition in [
                FixtureCondition::Near,
                FixtureCondition::Distance,
                FixtureCondition::Noise,
                FixtureCondition::Overlap,
            ] {
                let selected = fixtures
                    .iter()
                    .filter(|fixture| {
                        fixture.metadata.split == split && fixture.metadata.condition == condition
                    })
                    .collect::<Vec<_>>();
                reports.push(DimensionSliceReport {
                    split,
                    dimension: "condition",
                    value: condition.as_str(),
                    metrics: evaluate_selected(&selected, profile_policy, anonymous_policy),
                });
            }
        }
        reports
    }

    impl FixtureLanguage {
        fn as_str(self) -> &'static str {
            match self {
                Self::Chinese => "chinese",
                Self::English => "english",
                Self::Mixed => "mixed",
            }
        }
    }

    impl FixtureCondition {
        fn as_str(self) -> &'static str {
            match self {
                Self::Near => "near",
                Self::Distance => "distance",
                Self::Noise => "noise",
                Self::Overlap => "overlap",
            }
        }
    }

    fn evaluate_verification(
        fixtures: &[&EmbeddedFixture],
        policy: SpeakerMatchPolicy,
        report: &mut SplitReport,
    ) {
        for (probe_index, probe) in fixtures.iter().enumerate() {
            for candidate in fixtures.iter().skip(probe_index + 1) {
                let same_speaker = probe.metadata.speaker_id == candidate.metadata.speaker_id;
                let accepted = matches!(
                    match_speaker_profile(
                        &probe.embedding,
                        probe.quality,
                        &[SpeakerMatchCandidate {
                            profile_id: Uuid::new_v4(),
                            prototype: &candidate.embedding,
                        }],
                        policy,
                    ),
                    SpeakerMatchDecision::Matched { .. }
                );
                if same_speaker {
                    report.verification_same_speaker_trials += 1;
                    report.false_reject_count += usize::from(!accepted);
                } else {
                    report.verification_different_speaker_trials += 1;
                    report.false_accept_count += usize::from(accepted);
                }
            }
        }
    }

    fn evaluate_profiles(
        fixtures: &[&EmbeddedFixture],
        policy: SpeakerMatchPolicy,
        report: &mut SplitReport,
    ) {
        let speaker_ids = fixtures
            .iter()
            .map(|fixture| fixture.metadata.speaker_id.clone())
            .collect::<BTreeSet<_>>();
        let profile_ids = speaker_ids
            .into_iter()
            .map(|speaker_id| (speaker_id, Uuid::new_v4()))
            .collect::<BTreeMap<_, _>>();
        let candidates = fixtures
            .iter()
            .filter(|fixture| fixture.metadata.role == FixtureRole::Enrollment)
            .filter_map(|fixture| {
                profile_ids
                    .get(&fixture.metadata.speaker_id)
                    .map(|profile_id| SpeakerMatchCandidate {
                        profile_id: *profile_id,
                        prototype: &fixture.embedding,
                    })
            })
            .collect::<Vec<_>>();

        for probe in fixtures
            .iter()
            .filter(|fixture| fixture.metadata.role == FixtureRole::Probe)
        {
            report.profile_probe_count += 1;
            let expected = profile_ids.get(&probe.metadata.speaker_id);
            match match_speaker_profile(&probe.embedding, probe.quality, &candidates, policy) {
                SpeakerMatchDecision::Matched { profile_id, .. }
                    if expected == Some(&profile_id) =>
                {
                    report.correct_profile_count += 1;
                }
                SpeakerMatchDecision::Matched { .. } => report.incorrect_profile_count += 1,
                SpeakerMatchDecision::Unknown { .. }
                | SpeakerMatchDecision::Ambiguous { .. }
                | SpeakerMatchDecision::Ineligible(_) => report.uncertainty_count += 1,
            }
        }
    }

    fn evaluate_clustering(
        fixtures: &[&EmbeddedFixture],
        policy: SpeakerMatchPolicy,
        report: &mut SplitReport,
    ) {
        let mut sessions = BTreeMap::<&str, Vec<&EmbeddedFixture>>::new();
        for fixture in fixtures {
            sessions
                .entry(&fixture.metadata.session_id)
                .or_default()
                .push(fixture);
        }
        for session_fixtures in sessions.into_values() {
            let mut clusterer = SessionSpeakerClusterer::new(Uuid::new_v4(), policy)
                .expect("benchmark session IDs are non-empty");
            let assignments = session_fixtures
                .iter()
                .map(
                    |fixture| match clusterer.assign(fixture.embedding.clone(), fixture.quality) {
                        AnonymousSpeakerAssignment::Assigned { cluster_id, .. }
                        | AnonymousSpeakerAssignment::Created { cluster_id } => Some(cluster_id),
                        AnonymousSpeakerAssignment::Ambiguous
                        | AnonymousSpeakerAssignment::Ineligible(_)
                        | AnonymousSpeakerAssignment::IncompatibleModelSpace
                        | AnonymousSpeakerAssignment::CapacityReached => None,
                    },
                )
                .collect::<Vec<_>>();
            for left in 0..session_fixtures.len() {
                for right in (left + 1)..session_fixtures.len() {
                    report.anonymous_pair_trials += 1;
                    let truth_same = session_fixtures[left].metadata.speaker_id
                        == session_fixtures[right].metadata.speaker_id;
                    let predicted_same = assignments[left]
                        .as_ref()
                        .zip(assignments[right].as_ref())
                        .is_some_and(|(left, right)| left == right);
                    report.anonymous_pair_confusion_count +=
                        usize::from(truth_same != predicted_same);
                }
            }
        }
    }

    fn inference_report(
        fixtures: &[EmbeddedFixture],
        model_load_duration: Duration,
    ) -> InferenceReport {
        let mut latencies = fixtures
            .iter()
            .map(|fixture| fixture.inference_duration.as_secs_f64() * 1_000.0)
            .collect::<Vec<_>>();
        let mut realtime_factors = fixtures
            .iter()
            .map(|fixture| {
                fixture.inference_duration.as_secs_f64() / fixture.audio_duration.as_secs_f64()
            })
            .collect::<Vec<_>>();
        latencies.sort_by(f64::total_cmp);
        realtime_factors.sort_by(f64::total_cmp);
        InferenceReport {
            model_load_ms: model_load_duration.as_secs_f64() * 1_000.0,
            embedding_p50_ms: percentile(&latencies, 0.50),
            embedding_p95_ms: percentile(&latencies, 0.95),
            embedding_max_ms: latencies.last().copied().unwrap_or_default(),
            realtime_factor_p95: percentile(&realtime_factors, 0.95),
            peak_resident_memory_bytes: peak_resident_memory_bytes(),
            power_measurement:
                "external measurement required; use powermetrics during the same PID run",
        }
    }

    fn load_fixture_manifest(root: &Path) -> Result<FixtureManifest, String> {
        let path = resolve_fixture_file(root, Path::new("manifest.json"))?;
        let contents = fs::read_to_string(path)
            .map_err(|error| format!("fixture manifest is unreadable: {error}"))?;
        serde_json::from_str(&contents)
            .map_err(|error| format!("fixture manifest is invalid: {error}"))
    }

    fn validate_fixture_manifest(manifest: &FixtureManifest) -> Result<(), String> {
        if manifest.schema_version != 1 {
            return Err("fixture manifest schemaVersion must be 1".to_owned());
        }
        if manifest.corpus_id.trim().is_empty()
            || manifest.consent.statement.trim().is_empty()
            || manifest.consent.collected_at.trim().is_empty()
        {
            return Err("fixture corpus and consent metadata must not be empty".to_owned());
        }
        if !manifest.consent.confirmed {
            return Err("fixture manifest must record explicit local biometric consent".to_owned());
        }
        if manifest.fixtures.len() < 4 {
            return Err("fixture corpus must contain at least four clips".to_owned());
        }
        let mut ids = BTreeSet::new();
        let mut speakers = BTreeSet::new();
        for fixture in &manifest.fixtures {
            if fixture.id.trim().is_empty()
                || fixture.speaker_id.trim().is_empty()
                || fixture.session_id.trim().is_empty()
                || !ids.insert(&fixture.id)
            {
                return Err("fixture IDs, speaker IDs, and session IDs must be unique/non-empty as applicable".to_owned());
            }
            speakers.insert(&fixture.speaker_id);
        }
        if speakers.len() < 2 {
            return Err("fixture corpus must contain at least two consented speakers".to_owned());
        }
        for split in [FixtureSplit::Calibration, FixtureSplit::Acceptance] {
            let split_fixtures = manifest
                .fixtures
                .iter()
                .filter(|fixture| fixture.split == split)
                .collect::<Vec<_>>();
            if !split_fixtures
                .iter()
                .any(|fixture| fixture.role == FixtureRole::Enrollment)
                || !split_fixtures
                    .iter()
                    .any(|fixture| fixture.role == FixtureRole::Probe)
            {
                return Err(format!(
                    "{split:?} split must contain enrollment and probe fixtures"
                ));
            }
        }
        Ok(())
    }

    fn read_fixture_wav(root: &Path, relative: &Path) -> Result<Vec<f32>, String> {
        let path = resolve_fixture_file(root, relative)?;
        let mut reader = WavReader::open(&path).map_err(|error| {
            format!(
                "fixture {} is not a readable WAV: {error}",
                relative.display()
            )
        })?;
        let spec = reader.spec();
        if spec.channels != INFERENCE_CHANNELS || spec.sample_rate != INFERENCE_SAMPLE_RATE_HZ {
            return Err(format!(
                "fixture {} must be 16 kHz mono WAV",
                relative.display()
            ));
        }
        let samples = match spec.sample_format {
            SampleFormat::Float if spec.bits_per_sample == 32 => reader
                .samples::<f32>()
                .map(|sample| sample.map_err(|error| error.to_string()))
                .collect::<Result<Vec<_>, _>>()?,
            SampleFormat::Int if spec.bits_per_sample <= 16 => {
                let scale = f32::from(i16::MAX);
                reader
                    .samples::<i16>()
                    .map(|sample| {
                        sample
                            .map(|value| f32::from(value) / scale)
                            .map_err(|error| error.to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()?
            }
            _ => {
                return Err(format!(
                    "fixture {} must use 16-bit integer or 32-bit float PCM",
                    relative.display()
                ));
            }
        };
        if samples.is_empty() || samples.len() > MAX_INFERENCE_WINDOW_SAMPLES {
            return Err(format!(
                "fixture {} must contain 1..={MAX_INFERENCE_WINDOW_SAMPLES} samples",
                relative.display()
            ));
        }
        Ok(samples)
    }

    fn canonical_directory(path: &Path) -> Result<PathBuf, String> {
        if !path.is_absolute() {
            return Err("fixture/resource directory must be absolute".to_owned());
        }
        let metadata = fs::symlink_metadata(path)
            .map_err(|error| format!("fixture/resource directory is unavailable: {error}"))?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err("fixture/resource directory must be a real directory".to_owned());
        }
        fs::canonicalize(path)
            .map_err(|error| format!("fixture/resource directory is unavailable: {error}"))
    }

    fn resolve_fixture_file(root: &Path, relative: &Path) -> Result<PathBuf, String> {
        if relative.is_absolute()
            || relative.components().next().is_none()
            || !relative
                .components()
                .all(|component| matches!(component, Component::Normal(_)))
        {
            return Err("fixture path must be a relative normal path".to_owned());
        }
        let candidate = root.join(relative);
        let metadata = fs::symlink_metadata(&candidate)
            .map_err(|error| format!("fixture {} is unavailable: {error}", relative.display()))?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(format!(
                "fixture {} must be a regular non-symlink file",
                relative.display()
            ));
        }
        let canonical = fs::canonicalize(candidate)
            .map_err(|error| format!("fixture {} is unavailable: {error}", relative.display()))?;
        if !canonical.starts_with(root) {
            return Err("fixture path escaped the consented corpus directory".to_owned());
        }
        Ok(canonical)
    }

    fn unique_count<'a>(
        fixtures: &'a [EmbeddedFixture],
        select: impl Fn(&'a EmbeddedFixture) -> &'a String,
    ) -> usize {
        fixtures.iter().map(select).collect::<BTreeSet<_>>().len()
    }

    fn rate(numerator: usize, denominator: usize) -> Option<f64> {
        (denominator != 0).then_some(numerator as f64 / denominator as f64)
    }

    fn percentile(sorted: &[f64], percentile: f64) -> f64 {
        if sorted.is_empty() {
            return 0.0;
        }
        let index = ((sorted.len() - 1) as f64 * percentile).ceil() as usize;
        sorted[index]
    }

    #[cfg(target_os = "macos")]
    fn peak_resident_memory_bytes() -> Option<u64> {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
        // SAFETY: getrusage initializes the provided rusage when it returns zero.
        let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
        (result == 0).then(|| {
            // macOS reports ru_maxrss in bytes.
            u64::try_from(unsafe { usage.assume_init() }.ru_maxrss).unwrap_or_default()
        })
    }
}

#[cfg(target_os = "macos")]
fn main() {
    if let Err(error) = macos::main() {
        eprintln!("speaker recognition benchmark failed: {error}");
        std::process::exit(1);
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    println!("speaker recognition benchmark is available only on macOS");
}
