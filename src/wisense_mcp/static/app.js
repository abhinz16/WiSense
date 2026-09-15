"use strict";


const state = {
    overview: null,
    analytics: null,
    samples: [],
    selectedSample: null,
    externalFile: null,
    analysisSource: "dataset"
};


const pageTitles = {
    overview: "System Overview",
    analyze: "Live Analyze",
    samples: "Sample Explorer",
    analytics: "Model Analytics",
    diagnostics: "Sensing Diagnostics",
    model: "Model Architecture"
};


// ============================================================
// HELPERS
// ============================================================

/** Format a fractional metric as a percentage for display. */
function percent(value, digits = 1) {
    if (
        value === null
        || value === undefined
        || Number.isNaN(Number(value))
    ) {
        return "—";
    }

    return (
        (Number(value) * 100).toFixed(digits)
        + "%"
    );
}


/** Format a numeric value, or show an em dash when it is unavailable. */
function numericOrDash(value, digits = 3) {
    if (
        value === null
        || value === undefined
        || Number.isNaN(Number(value))
    ) {
        return "—";
    }

    return Number(value).toFixed(digits);
}


/** Convert a normalized score into a safe percentage width for progress bars. */
function barWidth(value) {
    if (
        value === null
        || value === undefined
        || Number.isNaN(Number(value))
    ) {
        return 0;
    }

    return Math.max(
        0,
        Math.min(
            100,
            Number(value) * 100
        )
    );
}


/** Escape text before placing it inside generated HTML. */
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
    if (
        value === null
        || value === undefined
    ) {
        return "—";
    }

    return String(value)
        .replaceAll("_", " ")
        .replace(
            /\b\w/g,
            character => character.toUpperCase()
        );
}


/** Fetch JSON from the dashboard API and raise useful errors for failed requests. */
async function api(url, options = {}) {
    const response = await fetch(url, options);

    const payload = await response.json();

    if (!response.ok) {
        throw new Error(
            payload.error
            || `HTTP ${response.status}`
        );
    }

    return payload;
}


/** Show a short status message without interrupting the workflow. */
function toast(message) {
    const element = document.getElementById("toast");

    element.textContent = message;
    element.classList.add("visible");

    setTimeout(
        () => {
            element.classList.remove("visible");
        },
        2200
    );
}


// ============================================================
// NAVIGATION
// ============================================================

/** Switch the visible dashboard page and keep the sidebar selection in sync. */
function navigate(page) {
    document
        .querySelectorAll(".page")
        .forEach(
            element => {
                element.classList.remove("active");
            }
        );

    document
        .querySelectorAll(".nav-item")
        .forEach(
            element => {
                element.classList.remove("active");
            }
        );

    const pageElement = document.getElementById(
        `page-${page}`
    );

    if (pageElement) {
        pageElement.classList.add("active");
    }

    const navElement = document.querySelector(
        `[data-page="${page}"]`
    );

    if (navElement) {
        navElement.classList.add("active");
    }

    document.getElementById("pageTitle").textContent = (
        pageTitles[page]
        || "WiSense"
    );
}


// ============================================================
// CONNECTION
// ============================================================

/** Refresh backend and GPU status indicators. */
async function refreshConnection() {
    const dot = document.getElementById("connectionDot");
    const gpuDot = document.getElementById("gpuDot");

    try {
        const health = await api("/api/health");

        dot.className = "status-dot status-online";

        document.getElementById(
            "connectionText"
        ).textContent = "WiSense Online";

        document.getElementById(
            "connectionDetail"
        ).textContent = "Persistent HTTP service";

        if (health.cuda_available) {
            gpuDot.className = "status-dot status-online";

            document.getElementById(
                "gpuText"
            ).textContent = (
                health.gpu
                || "CUDA Online"
            );
        } else {
            gpuDot.className = "status-dot status-waiting";

            document.getElementById(
                "gpuText"
            ).textContent = "CPU inference";
        }

    } catch (error) {
        dot.className = "status-dot status-offline";
        gpuDot.className = "status-dot status-offline";

        document.getElementById(
            "connectionText"
        ).textContent = "Reconnecting…";

        document.getElementById(
            "connectionDetail"
        ).textContent = "Server unavailable";

        document.getElementById(
            "gpuText"
        ).textContent = "Offline";
    }
}


// ============================================================
// OVERVIEW
// ============================================================

/** Load the dataset, final metrics, model status, and shared filter options. */
async function loadOverview() {
    const data = await api("/api/overview");

    state.overview = data;

    const metrics = data.model.metrics;

    document.getElementById(
        "metricPresence"
    ).textContent = percent(
        metrics.presence_macro_f1
    );

    document.getElementById(
        "metricActivity"
    ).textContent = percent(
        metrics.activity_macro_f1
    );

    document.getElementById(
        "metricStructured"
    ).textContent = percent(
        metrics.structured_macro_f1
    );

    document.getElementById(
        "metricSamples"
    ).textContent = Number(
        data.dataset.total_samples
    ).toLocaleString();

    document.getElementById(
        "datasetSummary"
    ).innerHTML = [
        [
            "Samples",
            Number(
                data.dataset.total_samples
            ).toLocaleString()
        ],
        [
            "WiFi Bands",
            data.dataset.wifi_bands_ghz
                .map(value => `${value} GHz`)
                .join(" / ")
        ],
        [
            "Activities",
            data.dataset.num_activities
        ],
        [
            "User Identities",
            data.dataset.num_users
        ],
        [
            "Environments",
            data.dataset.environments.length
        ],
        [
            "Input",
            (data.dataset.num_channels && data.dataset.target_length)
                ? `${data.dataset.num_channels} × ${data.dataset.target_length}`
                : "—"
        ]
    ].map(
        item => `
            <div class="detail-item">
                <span>${escapeHtml(item[0])}</span>
                <strong>${escapeHtml(item[1])}</strong>
            </div>
        `
    ).join("");

    const inference = data.inference;

    const epochValues = Object.values(
        inference.checkpoint_epochs || {}
    ).filter(value => value !== null && value !== undefined);

    document.getElementById(
        "systemSummary"
    ).innerHTML = [
        [
            "Device",
            String(inference.device || "cpu").toUpperCase()
        ],
        [
            "Production Model",
            "Temporal-Pyramid Ensemble"
        ],
        [
            "Ensemble",
            inference.ensemble_size ? `${inference.ensemble_size} seeds · equal weight` : "—"
        ],
        [
            "Checkpoints",
            epochValues.length
                ? epochValues.map(value => `E${value}`).join(" / ")
                : "Lazy load"
        ],
        [
            "Presence Threshold",
            Number(
                inference.presence_threshold
            ).toFixed(2)
        ],
        [
            "MCP",
            "/mcp"
        ]
    ].map(
        item => `
            <div class="detail-item">
                <span>${escapeHtml(item[0])}</span>
                <strong>${escapeHtml(item[1])}</strong>
            </div>
        `
    ).join("");

    populateFilters(
        data.sample_filters
    );

    configureExternalInput(
        data.external_input
    );
}


// ============================================================
// EXTERNAL SENSOR INPUT
// ============================================================

/** Configure the upload control from values exposed by config.ini. */
function configureExternalInput(config) {
    if (!config) {
        return;
    }

    const formats = Array.isArray(config.formats)
        ? config.formats
        : [];

    const input = document.getElementById(
        "externalSampleFile"
    );

    input.accept = formats.join(",");

    document.getElementById(
        "externalInputRequirement"
    ).textContent = (
        `${config.num_channels} CSI channels with the same antenna/subcarrier `
        + "ordering used during training."
    );

    document.getElementById(
        "externalInputFormats"
    ).textContent = (
        formats.length
            ? `Accepted: ${formats.join(", ")} · max ${config.max_upload_mb} MB`
            : ""
    );
}


/** Switch the live-analysis page between dataset and uploaded samples. */
function setAnalysisSource(source) {
    const normalized = source === "upload"
        ? "upload"
        : "dataset";

    state.analysisSource = normalized;

    document
        .querySelectorAll("[data-analysis-source]")
        .forEach(
            button => {
                button.classList.toggle(
                    "active",
                    button.dataset.analysisSource === normalized
                );
            }
        );

    document.getElementById(
        "datasetAnalysisSource"
    ).classList.toggle(
        "hidden",
        normalized !== "dataset"
    );

    document.getElementById(
        "uploadAnalysisSource"
    ).classList.toggle(
        "hidden",
        normalized !== "upload"
    );
}


/** Format an uploaded file size for the interface. */
function formatBytes(bytes) {
    const value = Number(bytes);

    if (!Number.isFinite(value) || value < 1024) {
        return `${Math.max(0, value || 0)} B`;
    }

    if (value < 1024 * 1024) {
        return `${(value / 1024).toFixed(1)} KB`;
    }

    return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}


/** Store a selected sensor file after basic browser-side validation. */
function setExternalFile(file) {
    const input = document.getElementById(
        "externalSampleFile"
    );
    const info = document.getElementById(
        "externalFileInfo"
    );
    const button = document.getElementById(
        "uploadAnalyzeButton"
    );

    if (!file) {
        state.externalFile = null;
        info.textContent = "No file selected";
        info.classList.remove("has-file");
        button.disabled = true;
        return;
    }

    const allowed = String(input.accept || "")
        .split(",")
        .map(value => value.trim().toLowerCase())
        .filter(Boolean);
    const lowerName = file.name.toLowerCase();
    const validExtension = (
        !allowed.length
        || allowed.some(extension => lowerName.endsWith(extension))
    );

    if (!validExtension) {
        setExternalFile(null);
        toast(`Unsupported file type. Use ${allowed.join(", ")}.`);
        return;
    }

    const limitMb = Number(
        state.overview?.external_input?.max_upload_mb
    );
    if (
        Number.isFinite(limitMb)
        && file.size > limitMb * 1024 * 1024
    ) {
        setExternalFile(null);
        toast(`Sample exceeds the ${limitMb} MB upload limit.`);
        return;
    }

    state.externalFile = file;
    info.textContent = `${file.name} · ${formatBytes(file.size)}`;
    info.classList.add("has-file");
    button.disabled = false;
}


// ============================================================
// FILTER POPULATION
// ============================================================

/** Add one option to a select control. */
function appendOption(select, value, label) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    select.appendChild(option);
}


/** Populate explorer and diagnostics filters from dataset metadata. */
function populateFilters(filters) {
    const environment = document.getElementById(
        "filterEnvironment"
    );

    const diagEnvironment = document.getElementById(
        "diagnosticEnvironment"
    );

    filters.environments.forEach(
        value => {
            const label = titleCase(value);
            appendOption(environment, value, label);
            appendOption(diagEnvironment, value, label);
        }
    );

    const band = document.getElementById(
        "filterBand"
    );

    const diagBand = document.getElementById(
        "diagnosticBand"
    );

    filters.bands.forEach(
        value => {
            const label = `${value} GHz`;
            appendOption(band, value, label);
            appendOption(diagBand, value, label);
        }
    );

    const users = document.getElementById(
        "filterUsers"
    );

    const diagUsers = document.getElementById(
        "diagnosticUsersFilter"
    );

    filters.user_counts.forEach(
        value => {
            const label = (
                value === 0
                    ? "No people"
                    : value === 1
                        ? "1 person"
                        : `${value} people`
            );

            appendOption(users, value, label);
            appendOption(diagUsers, value, label);
        }
    );
}


// ============================================================
// SAMPLE LIST
// ============================================================

/** Load the filtered sample list and wire each sample to live analysis. */
async function loadSamples() {
    const environment = document.getElementById(
        "filterEnvironment"
    ).value;

    const band = document.getElementById(
        "filterBand"
    ).value;

    const users = document.getElementById(
        "filterUsers"
    ).value;

    const search = document.getElementById(
        "filterSearch"
    ).value;

    const params = new URLSearchParams();
    params.set("limit", "60");

    if (environment) {
        params.set("environment", environment);
    }

    if (band) {
        params.set("band", band);
    }

    if (users) {
        params.set("users", users);
    }

    if (search) {
        params.set("search", search);
    }

    const result = await api(
        `/api/samples?${params}`
    );

    state.samples = result.samples;

    document.getElementById(
        "sampleCount"
    ).textContent = (
        `${result.total.toLocaleString()} `
        + (
            result.total === 1
                ? "sample"
                : "samples"
        )
        + " match"
    );

    const grid = document.getElementById(
        "sampleGrid"
    );

    grid.innerHTML = result.samples.map(
        sample => `
            <article class="sample-card">

                <div class="eyebrow">
                    ${escapeHtml(sample.environment_display)}
                </div>

                <h5>
                    ${escapeHtml(sample.display_name)}
                </h5>

                <div class="sample-meta">
                    <span class="meta-chip">
                        ${sample.wifi_band_ghz} GHz
                    </span>

                    <span class="meta-chip">
                        ${sample.number_of_users}
                        ${
                            sample.number_of_users === 1
                                ? "person"
                                : "people"
                        }
                    </span>
                </div>

                <button
                    class="button secondary sample-analyze"
                    data-alias="${escapeHtml(sample.alias)}"
                >
                    Analyze CSI
                </button>

            </article>
        `
    ).join("");

    document
        .querySelectorAll(".sample-analyze")
        .forEach(
            button => {
                button.addEventListener(
                    "click",
                    () => {
                        const alias = button.dataset.alias;

                        setAnalyzeSample(alias);
                        navigate("analyze");
                        runAnalysis(alias);
                    }
                );
            }
        );

    populateSampleSelects(
        result.samples
    );
}


/** Refresh the sample selectors used by analysis and diagnostics. */
function populateSampleSelects(samples) {
    const selects = [
        document.getElementById("analyzeSample"),
        document.getElementById("diagnosticSample")
    ].filter(Boolean);

    selects.forEach(
        select => {
            const previous = select.value;
            select.innerHTML = "";

            samples.forEach(
                sample => {
                    appendOption(
                        select,
                        sample.alias,
                        sample.display_name
                    );
                }
            );

            if (
                previous
                && samples.some(
                    sample => sample.alias === previous
                )
            ) {
                select.value = previous;
            }
        }
    );
}


/** Ensure a sample exists in a select control before choosing it. */
function ensureSelectOption(
    selectId,
    alias,
    displayName
) {
    const select = document.getElementById(selectId);

    const exists = [...select.options].some(
        option => option.value === alias
    );

    if (!exists) {
        appendOption(
            select,
            alias,
            displayName
        );
    }

    select.value = alias;
}


/** Select a sample in the live-analysis control. */
function setAnalyzeSample(alias) {
    const select = document.getElementById(
        "analyzeSample"
    );

    const existing = [...select.options].some(
        option => option.value === alias
    );

    if (existing) {
        select.value = alias;
    }
}


// ============================================================
// LIVE ANALYSIS
// ============================================================

/** Put one prediction result into the shared live-analysis result area. */
function showAnalysisResult(prediction, signalPoints, sourceLabel) {
    state.selectedSample = prediction;

    document.getElementById(
        "analysisHero"
    ).classList.remove("hidden");

    document.getElementById(
        "analysisSampleName"
    ).textContent = prediction.sample.display_name;

    document.getElementById(
        "analysisSource"
    ).textContent = sourceLabel;

    document.getElementById(
        "analysisLatency"
    ).textContent = (
        `${Number(prediction.inference_ms).toFixed(1)} ms`
    );

    document.getElementById(
        "analysisDevice"
    ).textContent = prediction.device.toUpperCase();

    drawSignal(signalPoints);
    renderUsers(prediction.users);
}


/** Run ensemble inference for a sample from the prepared WiMANS cache. */
async function runAnalysis(alias = null) {
    const select = document.getElementById(
        "analyzeSample"
    );

    alias = alias || select.value;

    if (!alias) {
        toast("Select a sample first.");
        return;
    }

    const button = document.getElementById(
        "analyzeButton"
    );

    button.disabled = true;
    button.textContent = "Running Ensemble…";

    try {
        const [prediction, signal] = await Promise.all([
            api(
                "/api/predict",
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        sample: alias,
                        split: "test",
                        top_k: 3
                    })
                }
            ),
            api(
                `/api/signal?sample=${encodeURIComponent(alias)}`
                + "&split=test&points=180"
            )
        ]);

        showAnalysisResult(
            prediction,
            signal.points,
            "DATASET"
        );

        toast("Live inference complete");
        await refreshConnection();

    } catch (error) {
        toast(error.message);

    } finally {
        button.disabled = false;
        button.textContent = "Run Neural Inference";
    }
}


/** Upload a compatible sensor capture and run the same production ensemble. */
async function runUploadedAnalysis() {
    const file = state.externalFile;
    if (!file) {
        toast("Choose a sensor sample first.");
        return;
    }

    const button = document.getElementById(
        "uploadAnalyzeButton"
    );

    button.disabled = true;
    button.textContent = "Uploading and Running…";

    try {
        const params = new URLSearchParams({
            filename: file.name,
            top_k: "3"
        });

        const prediction = await api(
            `/api/predict-upload?${params.toString()}`,
            {
                method: "POST",
                headers: {
                    "Content-Type": file.type || "application/octet-stream"
                },
                body: file
            }
        );

        showAnalysisResult(
            prediction,
            prediction.signal.points,
            "NEW SENSOR"
        );

        const details = prediction.preprocessing;
        const shape = details.original_shape.join(" × ");
        document.getElementById(
            "externalFileInfo"
        ).textContent = (
            `${file.name} · ${shape} · ${details.packets_used.toLocaleString()} packets used`
        );

        toast("Sensor sample prediction complete");
        await refreshConnection();

    } catch (error) {
        toast(error.message);

    } finally {
        button.disabled = false;
        button.textContent = "Run Neural Inference";
    }
}


/** Render per-user presence and activity predictions. */
function renderUsers(users) {
    const container = document.getElementById(
        "userResults"
    );

    container.innerHTML = users.map(
        user => {
            const status = user.predicted_present
                ? "present"
                : "absent";

            const activity = user.predicted_present
                ? user.predicted_activity
                : "No active user";

            const topRows = user.top_activities.map(
                activityItem => `
                    <div class="top-activity-row">

                        <span>
                            ${escapeHtml(
                                titleCase(activityItem.activity)
                            )}
                        </span>

                        <div class="probability-track">
                            <div
                                class="probability-fill"
                                style="width: ${barWidth(
                                    activityItem.probability
                                )}%"
                            ></div>
                        </div>

                        <span>
                            ${percent(
                                activityItem.probability,
                                0
                            )}
                        </span>

                    </div>
                `
            ).join("");

            return `
                <article class="user-card ${status}">

                    <div class="user-card-head">
                        <h5>User ${user.user_index}</h5>

                        <span class="presence-badge ${status}">
                            ${
                                user.predicted_present
                                    ? "PRESENT"
                                    : "ABSENT"
                            }
                        </span>
                    </div>

                    <div class="activity-name">
                        ${escapeHtml(titleCase(activity))}
                    </div>

                    <div class="activity-confidence">
                        Presence probability:
                        ${percent(user.presence_probability)}
                    </div>

                    <div class="top-activity">
                        ${topRows}
                    </div>

                </article>
            `;
        }
    ).join("");
}


// ============================================================
// SIGNAL CANVAS
// ============================================================

/** Draw the downsampled CSI energy trace on the analysis canvas. */
function drawSignal(points) {
    const canvas = document.getElementById(
        "signalCanvas"
    );

    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = 240;

    canvas.width = width * ratio;
    canvas.height = height * ratio;

    const context = canvas.getContext("2d");
    context.scale(ratio, ratio);

    context.clearRect(
        0,
        0,
        width,
        height
    );

    context.strokeStyle = "rgba(120,170,220,0.08)";
    context.lineWidth = 1;

    for (let y = 30; y < height; y += 30) {
        context.beginPath();
        context.moveTo(0, y);
        context.lineTo(width, y);
        context.stroke();
    }

    const gradient = context.createLinearGradient(
        0,
        0,
        width,
        0
    );

    gradient.addColorStop(0, "#54e8ff");
    gradient.addColorStop(1, "#9b7cff");

    context.strokeStyle = gradient;
    context.lineWidth = 2.3;
    context.shadowBlur = 10;
    context.shadowColor = "rgba(84,232,255,0.35)";

    context.beginPath();

    points.forEach(
        (point, index) => {
            const x = (
                index
                / (points.length - 1)
            ) * width;

            const y = (
                height - 18
            ) - (
                point.value
                * (height - 36)
            );

            if (index === 0) {
                context.moveTo(x, y);
            } else {
                context.lineTo(x, y);
            }
        }
    );

    context.stroke();
    context.shadowBlur = 0;
}


// ============================================================
// ANALYTICS
// ============================================================

/** Build reusable horizontal metric bars for analytics panels. */
function barRows(
    records,
    labelFunction,
    valueFunction
) {
    return records.map(
        record => {
            const value = Number(
                valueFunction(record)
            );

            if (Number.isNaN(value)) {
                return "";
            }

            return `
                <div class="chart-row">

                    <div class="chart-label">
                        ${escapeHtml(
                            labelFunction(record)
                        )}
                    </div>

                    <div class="chart-track">
                        <div
                            class="chart-fill"
                            style="width: ${barWidth(value)}%"
                        ></div>
                    </div>

                    <div class="chart-value">
                        ${percent(value)}
                    </div>

                </div>
            `;
        }
    ).join("");
}


/** Load final-test activity, subgroup, and confusion analytics. */
async function loadAnalytics() {
    const data = await api("/api/analytics");
    state.analytics = data;

    document.getElementById(
        "activityBars"
    ).innerHTML = barRows(
        [...data.activities].sort(
            (a, b) => b.f1 - a.f1
        ),
        record => titleCase(record.activity),
        record => record.f1
    );

    document.getElementById(
        "bandBars"
    ).innerHTML = barRows(
        data.bands,
        record => `${record.wifi_band_ghz} GHz`,
        record => record.activity_macro_f1
    );

    document.getElementById(
        "environmentBars"
    ).innerHTML = barRows(
        data.environments,
        record => titleCase(record.environment),
        record => record.activity_macro_f1
    );

    document.getElementById(
        "userCountBars"
    ).innerHTML = barRows(
        data.user_counts.filter(
            record => record.activity_macro_f1 !== null
        ),
        record => (
            record.number_of_users === 1
                ? "1 person"
                : `${record.number_of_users} people`
        ),
        record => record.activity_macro_f1
    );

    document.getElementById(
        "confusionList"
    ).innerHTML = data.confusions.map(
        confusion => `
            <div class="confusion-row">

                <strong>
                    ${escapeHtml(
                        titleCase(confusion.true_activity)
                    )}
                </strong>

                <div class="confusion-arrow">→</div>

                <strong>
                    ${escapeHtml(
                        titleCase(confusion.predicted_activity)
                    )}
                </strong>

                <div class="confusion-rate">
                    ${percent(
                        confusion.rate_given_true_activity
                    )}
                    ·
                    ${confusion.count} cases
                </div>

            </div>
        `
    ).join("");
}


// ============================================================
// MODEL
// ============================================================

/** Load architecture, checkpoint, ensemble, and locked-test details. */
async function loadModel() {
    const data = await api("/api/model");

    const config = data.model_config || {};
    const finalMetrics = (
        data.final_test
        && data.final_test.metrics
            ? data.final_test.metrics
            : {}
    );

    const epochs = Array.isArray(data.checkpoint_epochs)
        ? data.checkpoint_epochs
        : [];

    const seeds = (
        data.ensemble
        && Array.isArray(data.ensemble.seeds)
            ? data.ensemble.seeds
            : []
    );

    document.getElementById(
        "modelDetails"
    ).innerHTML = `
        <div class="panel-header">
            <div>
                <div class="eyebrow">
                    FROZEN PRODUCTION MODEL
                </div>
                <h4>${escapeHtml(data.name)}</h4>
            </div>
        </div>

        <div class="detail-grid">

            <div class="detail-item">
                <span>Checkpoint Epochs</span>
                <strong>
                    ${
                        epochs.length
                            ? epochs.map(
                                (epoch, index) =>
                                    `S${seeds[index] ?? index + 1}: E${epoch}`
                            ).join(" · ")
                            : "Loaded lazily"
                    }
                </strong>
            </div>

            <div class="detail-item">
                <span>Ensemble</span>
                <strong>
                    ${data.ensemble?.size ?? "—"} models ·
                    ${escapeHtml(
                        data.ensemble?.rule
                        || "equal-weight soft average"
                    )}
                </strong>
            </div>

            <div class="detail-item">
                <span>Presence Threshold</span>
                <strong>
                    ${Number(data.presence_threshold).toFixed(2)}
                </strong>
            </div>

            <div class="detail-item">
                <span>Length Aware</span>
                <strong>
                    True packet length + packed BiLSTM
                </strong>
            </div>

            <div class="detail-item">
                <span>CNN Channels</span>
                <strong>
                    ${config.base_channels ?? "—"}
                </strong>
            </div>

            <div class="detail-item">
                <span>CNN Kernel</span>
                <strong>
                    ${config.kernel_size ?? "—"}
                </strong>
            </div>

            <div class="detail-item">
                <span>LSTM Hidden</span>
                <strong>
                    ${config.lstm_hidden_size ?? "—"}
                    × 2 directions
                </strong>
            </div>

            <div class="detail-item">
                <span>LSTM Layers</span>
                <strong>
                    ${config.lstm_layers ?? "—"}
                </strong>
            </div>

            <div class="detail-item">
                <span>Temporal Pooling</span>
                <strong>
                    Global + early + middle + late
                </strong>
            </div>

            <div class="detail-item">
                <span>Inference Device</span>
                <strong>
                    ${escapeHtml(data.inference.device)}
                </strong>
            </div>

            <div class="detail-item">
                <span>Locked Test Presence F1</span>
                <strong>
                    ${percent(finalMetrics.presence_macro_f1)}
                </strong>
            </div>

            <div class="detail-item">
                <span>Locked Test Activity Top-1</span>
                <strong>
                    ${percent(
                        finalMetrics.activity_top1_accuracy
                    )}
                </strong>
            </div>

            <div class="detail-item">
                <span>Locked Test Activity F1</span>
                <strong>
                    ${percent(
                        finalMetrics.activity_macro_f1
                    )}
                </strong>
            </div>

            <div class="detail-item">
                <span>Locked Test Structured F1</span>
                <strong>
                    ${percent(finalMetrics.structured_macro_f1)}
                </strong>
            </div>

            <div class="detail-item">
                <span>Structured Accuracy</span>
                <strong>
                    ${percent(finalMetrics.structured_accuracy)}
                </strong>
            </div>

            <div class="detail-item">
                <span>Absent-user FPR</span>
                <strong>
                    ${percent(
                        finalMetrics.absent_user_fpr,
                        2
                    )}
                </strong>
            </div>

        </div>

        <div class="model-lock-note">
            <strong>
                Model locked before final test.
            </strong>
            Architecture, checkpoints, ensemble weights,
            preprocessing, and the configured presence threshold
            are frozen.
        </div>
    `;
}


// ============================================================
// SAMPLE DIAGNOSTICS
// ============================================================

/** Run signal-quality and prediction diagnostics for a selected sample. */
async function runDiagnostics(alias = null) {
    const select = document.getElementById(
        "diagnosticSample"
    );

    const sample = alias || select.value;

    if (!sample) {
        toast("Select a sample first.");
        return;
    }

    const button = document.getElementById(
        "diagnosticButton"
    );

    button.disabled = true;
    button.textContent = "Running Diagnostics…";

    try {
        const data = await api(
            "/api/diagnostics/sample",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    sample: sample,
                    split: "test"
                })
            }
        );

        document.getElementById(
            "diagnosticSummary"
        ).classList.remove("hidden");

        document.getElementById(
            "diagCoverage"
        ).textContent = percent(
            data.signal.packet_coverage
        );

        document.getElementById(
            "diagIntegrity"
        ).textContent = titleCase(
            data.integrity.integrity_status
        );

        document.getElementById(
            "diagTemporal"
        ).textContent = numericOrDash(
            data.signal.temporal_change_rms,
            3
        );

        document.getElementById(
            "diagLatency"
        ).textContent = (
            `${Number(
                data.prediction.inference_ms
            ).toFixed(1)} ms`
        );

        document.getElementById(
            "diagnosticNote"
        ).textContent = (
            data.signal.note
            + " "
            + data.prediction.activity_metric_note
        );

        renderDiagnosticUsers(
            data.prediction.users
        );

        toast("Diagnostics complete");

    } catch (error) {
        toast(error.message);

    } finally {
        button.disabled = false;
        button.textContent = "Diagnose Sample";
    }
}


/** Render ground-truth, prediction, and uncertainty details per user. */
function renderDiagnosticUsers(users) {
    const container = document.getElementById(
        "diagnosticUsers"
    );

    container.innerHTML = users.map(
        user => {
            const correct = user.structured_correct;

            const trueActivity = user.true_present
                ? user.true_activity
                : "Absent";

            const predictedActivity = user.predicted_present
                ? user.predicted_activity
                : "Absent";

            const activityMetrics = user.activity_metrics_applicable
                ? `
                    <div class="uncertainty-row">
                        <span>Activity entropy</span>

                        <div class="probability-track">
                            <div
                                class="probability-fill"
                                style="width: ${barWidth(
                                    user.activity_normalized_entropy
                                )}%"
                            ></div>
                        </div>

                        <span>
                            ${percent(
                                user.activity_normalized_entropy,
                                0
                            )}
                        </span>
                    </div>

                    <div class="uncertainty-row">
                        <span>Top-1 / Top-2 margin</span>

                        <div class="probability-track">
                            <div
                                class="probability-fill"
                                style="width: ${barWidth(
                                    user.activity_top1_top2_margin
                                )}%"
                            ></div>
                        </div>

                        <span>
                            ${percent(
                                user.activity_top1_top2_margin,
                                0
                            )}
                        </span>
                    </div>
                `
                : `
                    <div class="uncertainty-na">
                        Activity uncertainty is not scored for
                        this ground-truth absent user slot because
                        the activity head was not supervised on
                        absent-user rows.
                    </div>
                `;

            return `
                <article
                    class="diagnostic-user ${
                        correct
                            ? "correct"
                            : "incorrect"
                    }"
                >

                    <div class="diagnostic-user-header">

                        <strong>
                            User ${user.user_index}
                        </strong>

                        <span
                            class="${
                                correct
                                    ? "correct-badge"
                                    : "error-badge"
                            }"
                        >
                            ${
                                correct
                                    ? "CORRECT"
                                    : "ERROR"
                            }
                        </span>

                    </div>

                    <div class="diag-comparison">

                        <div class="diag-box">
                            <span>Ground Truth</span>
                            <strong>
                                ${escapeHtml(
                                    titleCase(trueActivity)
                                )}
                            </strong>
                        </div>

                        <div class="diag-box">
                            <span>Prediction</span>
                            <strong>
                                ${escapeHtml(
                                    titleCase(predictedActivity)
                                )}
                            </strong>
                        </div>

                    </div>

                    <div class="uncertainty-row">
                        <span>Presence probability</span>

                        <div class="probability-track">
                            <div
                                class="probability-fill"
                                style="width: ${barWidth(
                                    user.presence_probability
                                )}%"
                            ></div>
                        </div>

                        <span>
                            ${percent(
                                user.presence_probability,
                                0
                            )}
                        </span>
                    </div>

                    ${activityMetrics}

                </article>
            `;
        }
    ).join("");
}


// ============================================================
// ERROR / DIFFICULTY EXPLORER
// ============================================================

/** Build the shared query parameters for diagnostic sample searches. */
function diagnosticQueryParameters() {
    const params = new URLSearchParams();

    const environment = document.getElementById(
        "diagnosticEnvironment"
    ).value;

    const band = document.getElementById(
        "diagnosticBand"
    ).value;

    const users = document.getElementById(
        "diagnosticUsersFilter"
    ).value;

    params.set("limit", "30");

    if (environment) {
        params.set("environment", environment);
    }

    if (band) {
        params.set("band", band);
    }

    if (users) {
        params.set("users", users);
    }

    return params;
}


/** Find final-test samples that match the selected error category. */
async function loadModelErrors() {
    const params = diagnosticQueryParameters();

    params.set(
        "error_type",
        document.getElementById(
            "diagnosticErrorType"
        ).value
    );

    try {
        const result = await api(
            `/api/diagnostics/errors?${params}`
        );

        document.getElementById(
            "diagnosticResultCount"
        ).textContent = (
            `${result.matching_samples} matching error `
            + (
                result.matching_samples === 1
                    ? "sample"
                    : "samples"
            )
        );

        renderErrorSamples(
            result.samples
        );

    } catch (error) {
        toast(error.message);
    }
}


/** Render samples returned by the error explorer. */
function renderErrorSamples(samples) {
    const container = document.getElementById(
        "diagnosticErrorGrid"
    );

    container.innerHTML = samples.map(
        item => `
            <article class="sample-card">

                <div class="eyebrow">
                    MODEL ERROR
                </div>

                <h5>
                    ${escapeHtml(item.sample.display_name)}
                </h5>

                <div class="error-counts">

                    ${
                        item.activity_error_count
                            ? `
                                <span class="error-chip">
                                    ${item.activity_error_count}
                                    activity
                                </span>
                            `
                            : ""
                    }

                    ${
                        item.false_presence_count
                            ? `
                                <span class="error-chip">
                                    ${item.false_presence_count}
                                    false presence
                                </span>
                            `
                            : ""
                    }

                    ${
                        item.missed_presence_count
                            ? `
                                <span class="error-chip">
                                    ${item.missed_presence_count}
                                    missed
                                </span>
                            `
                            : ""
                    }

                    <span class="meta-chip">
                        ${item.structured_error_count}
                        structured errors
                    </span>

                </div>

                <button
                    class="button secondary diagnostic-open"
                    data-alias="${escapeHtml(item.sample.alias)}"
                    data-name="${escapeHtml(item.sample.display_name)}"
                >
                    Diagnose This Sample
                </button>

            </article>
        `
    ).join("");

    attachDiagnosticOpenButtons();
}


/** Find and render the highest-ranked retrospective uncertainty samples. */
async function loadDifficultSamples() {
    const params = diagnosticQueryParameters();

    try {
        const result = await api(
            `/api/diagnostics/difficult?${params}`
        );

        document.getElementById(
            "diagnosticResultCount"
        ).textContent = (
            `${result.returned_samples} highest-ranked `
            + "retrospective uncertainty samples"
        );

        const container = document.getElementById(
            "diagnosticErrorGrid"
        );

        container.innerHTML = result.samples.map(
            item => `
                <article class="sample-card">

                    <div class="eyebrow">
                        UNCERTAINTY RANKING
                    </div>

                    <h5>
                        ${escapeHtml(item.sample.display_name)}
                    </h5>

                    <div class="difficulty-score">
                        ${Number(
                            item.difficulty_score
                        ).toFixed(3)}
                    </div>

                    <div class="sample-meta">

                        <span class="meta-chip">
                            Entropy:
                            ${percent(
                                item.mean_activity_entropy,
                                0
                            )}
                        </span>

                        <span class="meta-chip">
                            Margin:
                            ${percent(
                                item.mean_activity_top1_top2_margin,
                                0
                            )}
                        </span>

                        <span class="meta-chip">
                            Presence uncertainty:
                            ${percent(
                                item.mean_presence_uncertainty,
                                0
                            )}
                        </span>

                        <span class="meta-chip">
                            ${item.structured_error_count}
                            structured errors
                        </span>

                    </div>

                    <button
                        class="button secondary diagnostic-open"
                        data-alias="${escapeHtml(item.sample.alias)}"
                        data-name="${escapeHtml(item.sample.display_name)}"
                    >
                        Diagnose This Sample
                    </button>

                </article>
            `
        ).join("");

        attachDiagnosticOpenButtons();

    } catch (error) {
        toast(error.message);
    }
}


/** Attach diagnostic actions to dynamically generated sample cards. */
function attachDiagnosticOpenButtons() {
    document
        .querySelectorAll(".diagnostic-open")
        .forEach(
            button => {
                button.addEventListener(
                    "click",
                    () => {
                        const alias = button.dataset.alias;
                        const displayName = button.dataset.name;

                        ensureSelectOption(
                            "diagnosticSample",
                            alias,
                            displayName
                        );

                        window.scrollTo({
                            top: 0,
                            behavior: "smooth"
                        });

                        runDiagnostics(alias);
                    }
                );
            }
        );
}


// ============================================================
// EVENTS
// ============================================================

document
    .querySelectorAll(".nav-item")
    .forEach(
        item => {
            item.addEventListener(
                "click",
                () => {
                    navigate(item.dataset.page);
                }
            );
        }
    );


document
    .querySelectorAll("[data-nav-target]")
    .forEach(
        item => {
            item.addEventListener(
                "click",
                () => {
                    navigate(item.dataset.navTarget);
                }
            );
        }
    );


document.getElementById(
    "heroAnalyzeButton"
).addEventListener(
    "click",
    () => navigate("analyze")
);


document
    .querySelectorAll("[data-analysis-source]")
    .forEach(
        button => {
            button.addEventListener(
                "click",
                () => setAnalysisSource(
                    button.dataset.analysisSource
                )
            );
        }
    );


document.getElementById(
    "analyzeButton"
).addEventListener(
    "click",
    () => runAnalysis()
);


document.getElementById(
    "uploadAnalyzeButton"
).addEventListener(
    "click",
    runUploadedAnalysis
);


document.getElementById(
    "externalSampleFile"
).addEventListener(
    "change",
    event => setExternalFile(
        event.target.files?.[0] || null
    )
);


const externalDropZone = document.getElementById(
    "externalDropZone"
);

["dragenter", "dragover"].forEach(
    eventName => {
        externalDropZone.addEventListener(
            eventName,
            event => {
                event.preventDefault();
                externalDropZone.classList.add("dragover");
            }
        );
    }
);

["dragleave", "drop"].forEach(
    eventName => {
        externalDropZone.addEventListener(
            eventName,
            event => {
                event.preventDefault();
                externalDropZone.classList.remove("dragover");
            }
        );
    }
);

externalDropZone.addEventListener(
    "drop",
    event => setExternalFile(
        event.dataTransfer?.files?.[0] || null
    )
);


document.getElementById(
    "applyFilters"
).addEventListener(
    "click",
    loadSamples
);


document.getElementById(
    "filterSearch"
).addEventListener(
    "keydown",
    event => {
        if (event.key === "Enter") {
            loadSamples();
        }
    }
);


document.getElementById(
    "diagnosticButton"
).addEventListener(
    "click",
    () => runDiagnostics()
);


document.getElementById(
    "findErrorsButton"
).addEventListener(
    "click",
    loadModelErrors
);


document.getElementById(
    "findDifficultButton"
).addEventListener(
    "click",
    loadDifficultSamples
);


// ============================================================
// START
// ============================================================

/** Load the dashboard and start periodic service-status checks. */
async function initialize() {
    await refreshConnection();

    try {
        // Load overview first because it populates filter controls.
        await loadOverview();

        await Promise.all([
            loadSamples(),
            loadAnalytics(),
            loadModel()
        ]);

    } catch (error) {
        toast(error.message);
    }

    setInterval(
        refreshConnection,
        5000
    );
}


initialize();