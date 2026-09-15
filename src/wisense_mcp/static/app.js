"use strict";

const state = {
    overview: null,
    analytics: null,
    samples: [],
    selectedSample: null,
};

const pageTitles = {
    overview: "System Overview",
    analyze: "Live Analyze",
    samples: "Sample Explorer",
    analytics: "Model Analytics",
    diagnostics: "Sensing Diagnostics",
    model: "Model Architecture",
};

/** Format a fraction as a percentage. */
function percent(value, digits = 1) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "—";
    }
    return `${(Number(value) * 100).toFixed(digits)}%`;
}

/** Format a numeric value or return an em dash when unavailable. */
function numberOrDash(value, digits = 3) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "—";
    }
    return Number(value).toFixed(digits);
}

/** Convert a normalized metric into a safe CSS bar width. */
function barWidth(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return 0;
    }
    return Math.max(0, Math.min(100, Number(value) * 100));
}

/** Escape user-visible strings before inserting them into HTML. */
function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

/** Convert underscore-separated labels into readable title case. */
function titleCase(value) {
    if (value === null || value === undefined) {
        return "—";
    }
    return String(value)
        .replaceAll("_", " ")
        .replace(/\b\w/g, character => character.toUpperCase());
}

/** Fetch JSON and surface API errors as JavaScript exceptions. */
async function api(url, options = {}) {
    const response = await fetch(url, options);
    const payload = await response.json();
    if (!response.ok) {
        throw new Error(payload.error || `HTTP ${response.status}`);
    }
    return payload;
}

/** Show a short non-blocking message in the bottom-right toast. */
function toast(message) {
    const element = document.getElementById("toast");
    element.textContent = message;
    element.classList.add("visible");
    setTimeout(() => element.classList.remove("visible"), 2200);
}

/** Switch the visible dashboard page and sidebar state. */
function navigate(page) {
    document.querySelectorAll(".page").forEach(element => element.classList.remove("active"));
    document.querySelectorAll(".nav-item").forEach(element => element.classList.remove("active"));

    document.getElementById(`page-${page}`)?.classList.add("active");
    document.querySelector(`[data-page="${page}"]`)?.classList.add("active");
    document.getElementById("pageTitle").textContent = pageTitles[page] || "WiSense";
}

/** Refresh backend and GPU connectivity indicators. */
async function refreshConnection() {
    const dot = document.getElementById("connectionDot");
    const gpuDot = document.getElementById("gpuDot");

    try {
        const health = await api("/api/health");
        dot.className = "status-dot status-online";
        document.getElementById("connectionText").textContent = "WiSense Online";
        document.getElementById("connectionDetail").textContent = "Persistent HTTP service";

        if (health.cuda_available) {
            gpuDot.className = "status-dot status-online";
            document.getElementById("gpuText").textContent = health.gpu || "CUDA Online";
        } else {
            gpuDot.className = "status-dot status-waiting";
            document.getElementById("gpuText").textContent = "CPU inference";
        }
    } catch (error) {
        dot.className = "status-dot status-offline";
        gpuDot.className = "status-dot status-offline";
        document.getElementById("connectionText").textContent = "Reconnecting…";
        document.getElementById("connectionDetail").textContent = "Server unavailable";
        document.getElementById("gpuText").textContent = "Offline";
    }
}

/** Append one option to a select element. */
function appendOption(select, value, label) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    select.appendChild(option);
}

/** Populate explorer and diagnostics filters from dataset metadata. */
function populateFilters(filters) {
    const environment = document.getElementById("filterEnvironment");
    const diagnosticEnvironment = document.getElementById("diagnosticEnvironment");
    const band = document.getElementById("filterBand");
    const diagnosticBand = document.getElementById("diagnosticBand");
    const users = document.getElementById("filterUsers");
    const diagnosticUsers = document.getElementById("diagnosticUsersFilter");

    filters.environments.forEach(value => {
        const label = titleCase(value);
        appendOption(environment, value, label);
        appendOption(diagnosticEnvironment, value, label);
    });

    filters.bands.forEach(value => {
        appendOption(band, value, `${value} GHz`);
        appendOption(diagnosticBand, value, `${value} GHz`);
    });

    filters.user_counts.forEach(value => {
        const label = value === 0 ? "No people" : value === 1 ? "1 person" : `${value} people`;
        appendOption(users, value, label);
        appendOption(diagnosticUsers, value, label);
    });
}

/** Load headline dataset, model, and system information. */
async function loadOverview() {
    const data = await api("/api/overview");
    state.overview = data;

    const metrics = data.model.metrics || {};
    document.getElementById("metricPresence").textContent = percent(metrics.presence_macro_f1);
    document.getElementById("metricActivity").textContent = percent(metrics.activity_macro_f1);
    document.getElementById("metricStructured").textContent = percent(metrics.structured_macro_f1);
    document.getElementById("metricSamples").textContent = data.dataset.total_samples
        ? Number(data.dataset.total_samples).toLocaleString()
        : "—";

    const datasetItems = [
        ["Samples", data.dataset.total_samples ? Number(data.dataset.total_samples).toLocaleString() : "—"],
        ["WiFi Bands", (data.dataset.wifi_bands_ghz || []).map(value => `${value} GHz`).join(" / ") || "—"],
        ["Activities", data.dataset.num_activities ?? "—"],
        ["User Identities", data.dataset.num_users ?? "—"],
        ["Environments", data.dataset.environments?.length ?? "—"],
        ["Input", data.dataset.num_channels && data.dataset.target_length ? `${data.dataset.num_channels} × ${data.dataset.target_length}` : "—"],
    ];

    document.getElementById("datasetSummary").innerHTML = datasetItems
        .map(([label, value]) => `
            <div class="detail-item">
                <span>${escapeHtml(label)}</span>
                <strong>${escapeHtml(value)}</strong>
            </div>
        `)
        .join("");

    const inference = data.inference;
    const epochs = Object.values(inference.checkpoint_epochs || {}).filter(value => value !== null);
    const systemItems = [
        ["Device", String(inference.device || "cpu").toUpperCase()],
        ["Production Model", "Temporal-Pyramid Ensemble"],
        ["Ensemble", `${inference.ensemble_size || 3} seeds · equal weight`],
        ["Checkpoints", epochs.length ? epochs.map(value => `E${value}`).join(" / ") : "Lazy load"],
        ["Presence Threshold", Number(inference.presence_threshold).toFixed(2)],
        ["MCP", "/mcp"],
    ];

    document.getElementById("systemSummary").innerHTML = systemItems
        .map(([label, value]) => `
            <div class="detail-item">
                <span>${escapeHtml(label)}</span>
                <strong>${escapeHtml(value)}</strong>
            </div>
        `)
        .join("");

    populateFilters(data.sample_filters);
}

/** Fill the live-analysis and diagnostics sample selectors. */
function populateSampleSelects(samples) {
    ["analyzeSample", "diagnosticSample"].forEach(id => {
        const select = document.getElementById(id);
        if (!select) {
            return;
        }
        const previous = select.value;
        select.innerHTML = "";
        samples.forEach(sample => appendOption(select, sample.alias, sample.display_name));
        if (previous && samples.some(sample => sample.alias === previous)) {
            select.value = previous;
        }
    });
}

/** Ensure a sample exists in a select control before selecting it. */
function ensureSelectOption(selectId, alias, displayName) {
    const select = document.getElementById(selectId);
    const exists = [...select.options].some(option => option.value === alias);
    if (!exists) {
        appendOption(select, alias, displayName);
    }
    select.value = alias;
}

/** Select an existing sample in the live-analysis control. */
function setAnalyzeSample(alias) {
    const select = document.getElementById("analyzeSample");
    if ([...select.options].some(option => option.value === alias)) {
        select.value = alias;
    }
}

/** Query the sample explorer and render matching cards. */
async function loadSamples() {
    const params = new URLSearchParams();
    params.set("limit", "60");

    const environment = document.getElementById("filterEnvironment").value;
    const band = document.getElementById("filterBand").value;
    const users = document.getElementById("filterUsers").value;
    const search = document.getElementById("filterSearch").value;
    if (environment) params.set("environment", environment);
    if (band) params.set("band", band);
    if (users) params.set("users", users);
    if (search) params.set("search", search);

    const result = await api(`/api/samples?${params}`);
    state.samples = result.samples;
    document.getElementById("sampleCount").textContent = `${result.total.toLocaleString()} ${result.total === 1 ? "sample" : "samples"} match`;

    document.getElementById("sampleGrid").innerHTML = result.samples
        .map(sample => `
            <article class="sample-card">
                <div class="eyebrow">${escapeHtml(sample.environment_display)}</div>
                <h5>${escapeHtml(sample.display_name)}</h5>
                <div class="sample-meta">
                    <span class="meta-chip">${sample.wifi_band_ghz} GHz</span>
                    <span class="meta-chip">${sample.number_of_users} ${sample.number_of_users === 1 ? "person" : "people"}</span>
                </div>
                <button class="button secondary sample-analyze" data-alias="${escapeHtml(sample.alias)}">Analyze CSI</button>
            </article>
        `)
        .join("");

    document.querySelectorAll(".sample-analyze").forEach(button => {
        button.addEventListener("click", () => {
            const alias = button.dataset.alias;
            setAnalyzeSample(alias);
            navigate("analyze");
            runAnalysis(alias);
        });
    });

    populateSampleSelects(result.samples);
}

/** Run live inference and load the temporal trace for one sample. */
async function runAnalysis(alias = null) {
    const select = document.getElementById("analyzeSample");
    const sampleAlias = alias || select.value;
    if (!sampleAlias) {
        toast("Select a sample first.");
        return;
    }

    const button = document.getElementById("analyzeButton");
    button.disabled = true;
    button.textContent = "Running Ensemble…";

    try {
        const [prediction, signal] = await Promise.all([
            api("/api/predict", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({sample: sampleAlias, split: "test"}),
            }),
            api(`/api/signal?sample=${encodeURIComponent(sampleAlias)}&split=test`),
        ]);

        state.selectedSample = prediction;
        document.getElementById("analysisHero").classList.remove("hidden");
        document.getElementById("analysisSampleName").textContent = prediction.sample.display_name;
        document.getElementById("analysisLatency").textContent = `${Number(prediction.inference_ms).toFixed(1)} ms`;
        document.getElementById("analysisDevice").textContent = prediction.device.toUpperCase();
        drawSignal(signal.points);
        renderUsers(prediction.users);
        toast("Live inference complete");
    } catch (error) {
        toast(error.message);
    } finally {
        button.disabled = false;
        button.textContent = "Run Neural Inference";
    }
}

/** Render per-user presence and top activity probabilities. */
function renderUsers(users) {
    document.getElementById("userResults").innerHTML = users
        .map(user => {
            const status = user.predicted_present ? "present" : "absent";
            const activity = user.predicted_present ? user.predicted_activity : "No active user";
            const topRows = user.top_activities
                .map(item => `
                    <div class="top-activity-row">
                        <span>${escapeHtml(titleCase(item.activity))}</span>
                        <div class="probability-track"><div class="probability-fill" style="width:${barWidth(item.probability)}%"></div></div>
                        <span>${percent(item.probability, 0)}</span>
                    </div>
                `)
                .join("");

            return `
                <article class="user-card ${status}">
                    <div class="user-card-head">
                        <h5>User ${user.user_index}</h5>
                        <span class="presence-badge ${status}">${user.predicted_present ? "PRESENT" : "ABSENT"}</span>
                    </div>
                    <div class="activity-name">${escapeHtml(titleCase(activity))}</div>
                    <div class="activity-confidence">Presence probability: ${percent(user.presence_probability)}</div>
                    <div class="top-activity">${topRows}</div>
                </article>
            `;
        })
        .join("");
}

/** Draw the normalized CSI RMS trace on the analysis canvas. */
function drawSignal(points) {
    const canvas = document.getElementById("signalCanvas");
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = 240;
    canvas.width = width * ratio;
    canvas.height = height * ratio;

    const context = canvas.getContext("2d");
    context.scale(ratio, ratio);
    context.clearRect(0, 0, width, height);
    context.strokeStyle = "rgba(120,170,220,0.08)";
    context.lineWidth = 1;
    for (let y = 30; y < height; y += 30) {
        context.beginPath();
        context.moveTo(0, y);
        context.lineTo(width, y);
        context.stroke();
    }

    const gradient = context.createLinearGradient(0, 0, width, 0);
    gradient.addColorStop(0, "#54e8ff");
    gradient.addColorStop(1, "#9b7cff");
    context.strokeStyle = gradient;
    context.lineWidth = 2.3;
    context.beginPath();
    points.forEach((point, index) => {
        const x = points.length > 1 ? (index / (points.length - 1)) * width : 0;
        const y = height - 18 - point.value * (height - 36);
        if (index === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
    });
    context.stroke();
}

/** Build horizontal metric bars for analytics panels. */
function barRows(records, labelFunction, valueFunction) {
    return records
        .map(record => {
            const value = Number(valueFunction(record));
            if (Number.isNaN(value)) return "";
            return `
                <div class="chart-row">
                    <div class="chart-label">${escapeHtml(labelFunction(record))}</div>
                    <div class="chart-track"><div class="chart-fill" style="width:${barWidth(value)}%"></div></div>
                    <div class="chart-value">${percent(value)}</div>
                </div>
            `;
        })
        .join("");
}

/** Load final-test analytics into the performance page. */
async function loadAnalytics() {
    const data = await api("/api/analytics");
    state.analytics = data;

    document.getElementById("activityBars").innerHTML = barRows(
        [...data.activities].sort((a, b) => b.f1 - a.f1),
        record => titleCase(record.activity),
        record => record.f1,
    );
    document.getElementById("bandBars").innerHTML = barRows(
        data.bands,
        record => `${record.wifi_band_ghz} GHz`,
        record => record.activity_macro_f1,
    );
    document.getElementById("environmentBars").innerHTML = barRows(
        data.environments,
        record => titleCase(record.environment),
        record => record.activity_macro_f1,
    );
    document.getElementById("userCountBars").innerHTML = barRows(
        data.user_counts,
        record => record.number_of_users === 1 ? "1 person" : `${record.number_of_users} people`,
        record => record.activity_macro_f1,
    );
    document.getElementById("confusionList").innerHTML = data.confusions
        .map(confusion => `
            <div class="confusion-row">
                <strong>${escapeHtml(titleCase(confusion.true_activity))}</strong>
                <div class="confusion-arrow">→</div>
                <strong>${escapeHtml(titleCase(confusion.predicted_activity))}</strong>
                <div class="confusion-rate">${percent(confusion.rate_given_true_activity)} · ${confusion.count} cases</div>
            </div>
        `)
        .join("");
}

/** Load architecture and frozen-result details for the model page. */
async function loadModel() {
    const data = await api("/api/model");
    const config = data.model_config || {};
    const metrics = data.final_test?.metrics || {};
    const epochs = (data.checkpoint_epochs || []).filter(value => value !== null);

    document.getElementById("modelDetails").innerHTML = `
        <div class="panel-header">
            <div>
                <div class="eyebrow">FROZEN PRODUCTION MODEL</div>
                <h4>${escapeHtml(data.name)}</h4>
            </div>
        </div>
        <div class="detail-grid">
            <div class="detail-item"><span>Checkpoint Epochs</span><strong>${epochs.length ? epochs.map(epoch => `E${epoch}`).join(" · ") : "Loaded lazily"}</strong></div>
            <div class="detail-item"><span>Ensemble</span><strong>${data.ensemble.size} models · equal weight</strong></div>
            <div class="detail-item"><span>Presence Threshold</span><strong>${Number(data.presence_threshold).toFixed(2)}</strong></div>
            <div class="detail-item"><span>Length Aware</span><strong>True packet length + packed BiLSTM</strong></div>
            <div class="detail-item"><span>CNN Base Channels</span><strong>${config.base_channels ?? "—"}</strong></div>
            <div class="detail-item"><span>CNN Kernel</span><strong>${config.kernel_size ?? "—"}</strong></div>
            <div class="detail-item"><span>LSTM Hidden</span><strong>${config.lstm_hidden_size ?? "—"}</strong></div>
            <div class="detail-item"><span>LSTM Layers</span><strong>${config.lstm_layers ?? "—"}</strong></div>
            <div class="detail-item"><span>Temporal Pooling</span><strong>Global + coarse temporal segments</strong></div>
            <div class="detail-item"><span>Inference Device</span><strong>${escapeHtml(data.inference.device)}</strong></div>
            <div class="detail-item"><span>Test Presence F1</span><strong>${percent(metrics.presence_macro_f1)}</strong></div>
            <div class="detail-item"><span>Test Activity Top-1</span><strong>${percent(metrics.activity_top1_accuracy)}</strong></div>
            <div class="detail-item"><span>Test Activity F1</span><strong>${percent(metrics.activity_macro_f1)}</strong></div>
            <div class="detail-item"><span>Test Structured F1</span><strong>${percent(metrics.structured_macro_f1)}</strong></div>
            <div class="detail-item"><span>Structured Accuracy</span><strong>${percent(metrics.structured_accuracy)}</strong></div>
            <div class="detail-item"><span>Absent-user FPR</span><strong>${percent(metrics.absent_user_fpr, 2)}</strong></div>
        </div>
        <div class="model-lock-note">
            <strong>Model locked before final test.</strong>
            The final test set is reporting-only and is not used to adjust thresholds, preprocessing, checkpoints, or ensemble weights.
        </div>
    `;
}

/** Run CSI integrity and prediction diagnostics for one sample. */
async function runDiagnostics(alias = null) {
    const sample = alias || document.getElementById("diagnosticSample").value;
    if (!sample) {
        toast("Select a sample first.");
        return;
    }

    const button = document.getElementById("diagnosticButton");
    button.disabled = true;
    button.textContent = "Running Diagnostics…";

    try {
        const data = await api("/api/diagnostics/sample", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({sample, split: "test"}),
        });
        document.getElementById("diagnosticSummary").classList.remove("hidden");
        document.getElementById("diagCoverage").textContent = percent(data.signal.packet_coverage);
        document.getElementById("diagIntegrity").textContent = titleCase(data.integrity.integrity_status);
        document.getElementById("diagTemporal").textContent = numberOrDash(data.signal.temporal_change_rms, 3);
        document.getElementById("diagLatency").textContent = `${Number(data.prediction.inference_ms).toFixed(1)} ms`;
        document.getElementById("diagnosticNote").textContent = `${data.signal.note} ${data.prediction.activity_metric_note}`;
        renderDiagnosticUsers(data.prediction.users);
        toast("Diagnostics complete");
    } catch (error) {
        toast(error.message);
    } finally {
        button.disabled = false;
        button.textContent = "Diagnose Sample";
    }
}

/** Render user-level ground-truth, prediction, and uncertainty diagnostics. */
function renderDiagnosticUsers(users) {
    document.getElementById("diagnosticUsers").innerHTML = users
        .map(user => {
            const trueActivity = user.true_present ? user.true_activity : "Absent";
            const predictedActivity = user.predicted_present ? user.predicted_activity : "Absent";
            const uncertainty = user.activity_metrics_applicable
                ? `
                    <div class="uncertainty-row"><span>Activity entropy</span><div class="probability-track"><div class="probability-fill" style="width:${barWidth(user.activity_normalized_entropy)}%"></div></div><span>${percent(user.activity_normalized_entropy, 0)}</span></div>
                    <div class="uncertainty-row"><span>Top-1 / Top-2 margin</span><div class="probability-track"><div class="probability-fill" style="width:${barWidth(user.activity_top1_top2_margin)}%"></div></div><span>${percent(user.activity_top1_top2_margin, 0)}</span></div>
                `
                : `<div class="uncertainty-na">Activity uncertainty is not scored for a ground-truth absent user slot.</div>`;

            return `
                <article class="diagnostic-user ${user.structured_correct ? "correct" : "incorrect"}">
                    <div class="diagnostic-user-header"><strong>User ${user.user_index}</strong><span class="${user.structured_correct ? "correct-badge" : "error-badge"}">${user.structured_correct ? "CORRECT" : "ERROR"}</span></div>
                    <div class="diag-comparison">
                        <div class="diag-box"><span>Ground Truth</span><strong>${escapeHtml(titleCase(trueActivity))}</strong></div>
                        <div class="diag-box"><span>Prediction</span><strong>${escapeHtml(titleCase(predictedActivity))}</strong></div>
                    </div>
                    <div class="uncertainty-row"><span>Presence probability</span><div class="probability-track"><div class="probability-fill" style="width:${barWidth(user.presence_probability)}%"></div></div><span>${percent(user.presence_probability, 0)}</span></div>
                    ${uncertainty}
                </article>
            `;
        })
        .join("");
}

/** Build common diagnostic explorer query parameters. */
function diagnosticQueryParameters() {
    const params = new URLSearchParams();
    params.set("limit", "30");
    const environment = document.getElementById("diagnosticEnvironment").value;
    const band = document.getElementById("diagnosticBand").value;
    const users = document.getElementById("diagnosticUsersFilter").value;
    if (environment) params.set("environment", environment);
    if (band) params.set("band", band);
    if (users) params.set("users", users);
    return params;
}

/** Render frozen error-search results. */
function renderErrorSamples(samples) {
    document.getElementById("diagnosticErrorGrid").innerHTML = samples
        .map(item => `
            <article class="sample-card">
                <div class="eyebrow">MODEL ERROR</div>
                <h5>${escapeHtml(item.sample.display_name)}</h5>
                <div class="error-counts">
                    ${item.activity_error_count ? `<span class="error-chip">${item.activity_error_count} activity</span>` : ""}
                    ${item.false_presence_count ? `<span class="error-chip">${item.false_presence_count} false presence</span>` : ""}
                    ${item.missed_presence_count ? `<span class="error-chip">${item.missed_presence_count} missed</span>` : ""}
                    <span class="meta-chip">${item.structured_error_count} structured errors</span>
                </div>
                <button class="button secondary diagnostic-open" data-alias="${escapeHtml(item.sample.alias)}" data-name="${escapeHtml(item.sample.display_name)}">Diagnose This Sample</button>
            </article>
        `)
        .join("");
    attachDiagnosticOpenButtons();
}

/** Fetch samples containing the selected final-test error type. */
async function loadModelErrors() {
    const params = diagnosticQueryParameters();
    params.set("error_type", document.getElementById("diagnosticErrorType").value);
    try {
        const result = await api(`/api/diagnostics/errors?${params}`);
        document.getElementById("diagnosticResultCount").textContent = `${result.matching_samples} matching error ${result.matching_samples === 1 ? "sample" : "samples"}`;
        renderErrorSamples(result.samples);
    } catch (error) {
        toast(error.message);
    }
}

/** Fetch and render the highest-ranked retrospective uncertainty samples. */
async function loadDifficultSamples() {
    const params = diagnosticQueryParameters();
    try {
        const result = await api(`/api/diagnostics/difficult?${params}`);
        document.getElementById("diagnosticResultCount").textContent = `${result.returned_samples} highest-ranked retrospective uncertainty samples`;
        document.getElementById("diagnosticErrorGrid").innerHTML = result.samples
            .map(item => `
                <article class="sample-card">
                    <div class="eyebrow">UNCERTAINTY RANKING</div>
                    <h5>${escapeHtml(item.sample.display_name)}</h5>
                    <div class="difficulty-score">${Number(item.difficulty_score).toFixed(3)}</div>
                    <div class="sample-meta">
                        <span class="meta-chip">Entropy: ${percent(item.mean_activity_entropy, 0)}</span>
                        <span class="meta-chip">Margin: ${percent(item.mean_activity_top1_top2_margin, 0)}</span>
                        <span class="meta-chip">Presence uncertainty: ${percent(item.mean_presence_uncertainty, 0)}</span>
                        <span class="meta-chip">${item.structured_error_count} structured errors</span>
                    </div>
                    <button class="button secondary diagnostic-open" data-alias="${escapeHtml(item.sample.alias)}" data-name="${escapeHtml(item.sample.display_name)}">Diagnose This Sample</button>
                </article>
            `)
            .join("");
        attachDiagnosticOpenButtons();
    } catch (error) {
        toast(error.message);
    }
}

/** Attach diagnostic buttons after dynamically rendering sample cards. */
function attachDiagnosticOpenButtons() {
    document.querySelectorAll(".diagnostic-open").forEach(button => {
        button.addEventListener("click", () => {
            ensureSelectOption("diagnosticSample", button.dataset.alias, button.dataset.name);
            window.scrollTo({top: 0, behavior: "smooth"});
            runDiagnostics(button.dataset.alias);
        });
    });
}

/** Register static dashboard event handlers. */
function registerEvents() {
    document.querySelectorAll(".nav-item").forEach(item => {
        item.addEventListener("click", () => navigate(item.dataset.page));
    });
    document.querySelectorAll("[data-nav-target]").forEach(item => {
        item.addEventListener("click", () => navigate(item.dataset.navTarget));
    });
    document.getElementById("heroAnalyzeButton").addEventListener("click", () => navigate("analyze"));
    document.getElementById("analyzeButton").addEventListener("click", () => runAnalysis());
    document.getElementById("applyFilters").addEventListener("click", loadSamples);
    document.getElementById("filterSearch").addEventListener("keydown", event => {
        if (event.key === "Enter") loadSamples();
    });
    document.getElementById("diagnosticButton").addEventListener("click", () => runDiagnostics());
    document.getElementById("findErrorsButton").addEventListener("click", loadModelErrors);
    document.getElementById("findDifficultButton").addEventListener("click", loadDifficultSamples);
}

/** Load the initial dashboard data and start connection polling. */
async function initialize() {
    registerEvents();
    await refreshConnection();
    try {
        await loadOverview();
        await Promise.all([loadSamples(), loadAnalytics(), loadModel()]);
    } catch (error) {
        toast(error.message);
    }
    setInterval(refreshConnection, 5000);
}

initialize();
