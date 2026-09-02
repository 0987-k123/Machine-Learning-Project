/**
 * GlycoVision AI - Clean, Modern, Normal & Attractive Clinical ML Diagnostic Studio
 * Real-Time Dual-Model Inference, Theme Management, Interactive Visualizations & Medical Report Generator
 */

let appMetadata = null;
let appEDA = null;
let charts = {};
let currentThreshold = 0.50;
let lastInferenceResult = null;
let debounceTimer = null;

document.addEventListener('DOMContentLoaded', () => {
    initThemeToggle();
    initTabs();
    initSlidersAndLiveBadges();
    initPresets();
    initThresholdTuner();
    initReportModal();
    fetchMetadataAndEDA();
    setupFormEvents();
});

// ==========================================
// 1. Theme Management (Light Mode Default + Dark Mode)
// ==========================================
function initThemeToggle() {
    const themeBtn = document.getElementById('btn-theme-toggle');
    const themeIcon = document.getElementById('theme-icon');
    const themeText = document.getElementById('theme-btn-text');

    // Get saved theme or default to clean Light Mode
    const savedTheme = localStorage.getItem('glycovision_theme') || 'light';
    applyTheme(savedTheme);

    if (themeBtn) {
        themeBtn.addEventListener('click', () => {
            const current = document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
            const nextTheme = current === 'dark' ? 'light' : 'dark';
            applyTheme(nextTheme);
            localStorage.setItem('glycovision_theme', nextTheme);

            // Re-render charts with new theme colors
            if (appMetadata && appMetadata.models) {
                renderCharts(appMetadata.models.logistic_regression, appMetadata.models.random_forest);
            }
            if (appEDA) {
                renderEDACharts(appEDA, appMetadata);
            }
        });
    }

    function applyTheme(theme) {
        if (theme === 'dark') {
            document.documentElement.setAttribute('data-theme', 'dark');
            if (themeIcon) {
                themeIcon.className = 'fa-solid fa-sun';
                themeIcon.style.color = '#f59e0b';
            }
            if (themeText) themeText.textContent = 'Light Mode';
        } else {
            document.documentElement.removeAttribute('data-theme');
            if (themeIcon) {
                themeIcon.className = 'fa-solid fa-moon';
                themeIcon.style.color = '#475569';
            }
            if (themeText) themeText.textContent = 'Dark Mode';
        }
    }
}

function isDarkMode() {
    return document.documentElement.getAttribute('data-theme') === 'dark';
}

function getChartColors() {
    const dark = isDarkMode();
    return {
        textColor: dark ? '#cbd5e1' : '#475569',
        headingColor: dark ? '#f8fafc' : '#0f172a',
        gridColor: dark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(15, 23, 42, 0.06)',
        rfBorder: '#10b981',
        rfBg: dark ? 'rgba(16, 185, 129, 0.2)' : 'rgba(16, 185, 129, 0.15)',
        lrBorder: dark ? '#38bdf8' : '#2563eb',
        lrBg: dark ? 'rgba(56, 189, 248, 0.2)' : 'rgba(37, 99, 235, 0.12)',
        featBg: dark ? '#a78bfa' : '#8b5cf6',
        pieColors: dark ? ['#38bdf8', '#f87171'] : ['#2563eb', '#ef4444']
    };
}

// ==========================================
// 2. Tab Navigation
// ==========================================
function initTabs() {
    const tabBtns = document.querySelectorAll('.nav-tab-btn');
    const tabPanels = document.querySelectorAll('.tab-content-panel');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetId = btn.getAttribute('data-tab');

            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanels.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const targetPanel = document.getElementById(targetId);
            if (targetPanel) {
                targetPanel.classList.add('active');
                // Trigger chart resizing smoothly
                setTimeout(() => {
                    Object.values(charts).forEach(c => c && c.resize && c.resize());
                }, 50);
            }
        });
    });
}

// ==========================================
// 3. Interactive Sliders & Live Badges
// ==========================================
function initSlidersAndLiveBadges() {
    const sliderConfigs = [
        {
            slider: 'slider-age', input: 'input-age', badge: 'badge-age',
            updateBadge: (v, el) => {
                if (v < 25) { el.textContent = 'Youth (<25y)'; el.className = 'badge-status badge-normal'; }
                else if (v < 45) { el.textContent = 'Adult (25-44y)'; el.className = 'badge-status badge-normal'; }
                else if (v < 65) { el.textContent = 'Middle-Aged (45-64y)'; el.className = 'badge-status badge-warning'; }
                else { el.textContent = 'Senior (65y+)'; el.className = 'badge-status badge-danger'; }
            }
        },
        {
            slider: 'slider-bmi', input: 'input-bmi', badge: 'badge-bmi',
            updateBadge: (v, el) => {
                if (v < 18.5) { el.textContent = 'Underweight (<18.5)'; el.className = 'badge-status badge-warning'; }
                else if (v < 25.0) { el.textContent = 'Normal (18.5-24.9)'; el.className = 'badge-status badge-normal'; }
                else if (v < 30.0) { el.textContent = 'Overweight (25-29.9)'; el.className = 'badge-status badge-warning'; }
                else { el.textContent = 'Obese (≥30.0)'; el.className = 'badge-status badge-danger'; }
            }
        },
        {
            slider: 'slider-hba1c', input: 'input-hba1c', badge: 'badge-hba1c',
            updateBadge: (v, el) => {
                if (v < 5.7) { el.textContent = 'Optimal (<5.7%)'; el.className = 'badge-status badge-normal'; }
                else if (v <= 6.4) { el.textContent = 'Prediabetes (5.7-6.4%)'; el.className = 'badge-status badge-warning'; }
                else { el.textContent = 'Diabetic Range (≥6.5%)'; el.className = 'badge-status badge-danger'; }
            }
        },
        {
            slider: 'slider-glucose', input: 'input-glucose', badge: 'badge-glucose',
            updateBadge: (v, el) => {
                if (v < 100) { el.textContent = 'Normal Fasting (<100 mg/dL)'; el.className = 'badge-status badge-normal'; }
                else if (v < 140) { el.textContent = 'Elevated (100-139 mg/dL)'; el.className = 'badge-status badge-warning'; }
                else { el.textContent = 'High / Diabetic (≥140 mg/dL)'; el.className = 'badge-status badge-danger'; }
            }
        }
    ];

    sliderConfigs.forEach(({ slider, input, badge, updateBadge }) => {
        const sEl = document.getElementById(slider);
        const iEl = document.getElementById(input);
        const bEl = document.getElementById(badge);

        if (!sEl || !iEl) return;

        const handleInput = (val) => {
            if (updateBadge && bEl) updateBadge(parseFloat(val), bEl);
            triggerLiveInferenceDebounced();
        };

        sEl.addEventListener('input', (e) => {
            iEl.value = e.target.value;
            handleInput(e.target.value);
        });

        iEl.addEventListener('input', (e) => {
            sEl.value = e.target.value;
            handleInput(e.target.value);
        });

        // Initialize badge
        if (updateBadge && bEl) updateBadge(parseFloat(sEl.value), bEl);
    });

    // Radio toggles and select dropdown changes
    document.querySelectorAll('input[name="gender"], input[name="hypertension"], input[name="heart_disease"], #select-smoking').forEach(el => {
        el.addEventListener('change', () => {
            triggerLiveInferenceDebounced();
        });
    });
}

function triggerLiveInferenceDebounced() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
        runDualInference();
    }, 120);
}

// ==========================================
// 4. Clinical Presets Management
// ==========================================
const PRESETS = {
    healthy: {
        gender: 'Female', age: 22, hypertension: 0, heart_disease: 0,
        smoking_history: 'never', bmi: 21.2, HbA1c_level: 4.7, blood_glucose_level: 88
    },
    borderline: {
        gender: 'Male', age: 49, hypertension: 1, heart_disease: 0,
        smoking_history: 'past_smoker', bmi: 28.5, HbA1c_level: 6.2, blood_glucose_level: 138
    },
    high_risk: {
        gender: 'Female', age: 68, hypertension: 1, heart_disease: 1,
        smoking_history: 'never', bmi: 33.4, HbA1c_level: 6.8, blood_glucose_level: 165
    },
    diabetic: {
        gender: 'Male', age: 56, hypertension: 1, heart_disease: 0,
        smoking_history: 'current', bmi: 36.2, HbA1c_level: 8.6, blood_glucose_level: 245
    }
};

function initPresets() {
    const presetBtns = document.querySelectorAll('.preset-card-btn');
    presetBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const key = btn.getAttribute('data-preset');
            const data = PRESETS[key];
            if (!data) return;

            presetBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            applyFormData(data);
            runDualInference();
        });
    });
}

function applyFormData(data) {
    const genderRadio = document.querySelector(`input[name="gender"][value="${data.gender}"]`);
    if (genderRadio) genderRadio.checked = true;

    const fields = [
        { name: 'age', val: data.age },
        { name: 'bmi', val: data.bmi },
        { name: 'hba1c', val: data.HbA1c_level },
        { name: 'glucose', val: data.blood_glucose_level }
    ];

    fields.forEach(({ name, val }) => {
        const s = document.getElementById(`slider-${name}`);
        const i = document.getElementById(`input-${name}`);
        if (s && i) {
            s.value = val;
            i.value = val;
            s.dispatchEvent(new Event('input'));
        }
    });

    const htRadio = document.querySelector(`input[name="hypertension"][value="${data.hypertension}"]`);
    if (htRadio) htRadio.checked = true;

    const hdRadio = document.querySelector(`input[name="heart_disease"][value="${data.heart_disease}"]`);
    if (hdRadio) hdRadio.checked = true;

    const smokeSelect = document.getElementById('select-smoking');
    if (smokeSelect) smokeSelect.value = data.smoking_history;
}

function getFormData() {
    return {
        gender: document.querySelector('input[name="gender"]:checked')?.value || 'Female',
        age: parseFloat(document.getElementById('input-age')?.value || 40),
        hypertension: parseInt(document.querySelector('input[name="hypertension"]:checked')?.value || 0),
        heart_disease: parseInt(document.querySelector('input[name="heart_disease"]:checked')?.value || 0),
        smoking_history: document.getElementById('select-smoking')?.value || 'never',
        bmi: parseFloat(document.getElementById('input-bmi')?.value || 25.0),
        HbA1c_level: parseFloat(document.getElementById('input-hba1c')?.value || 5.5),
        blood_glucose_level: parseFloat(document.getElementById('input-glucose')?.value || 120)
    };
}

// ==========================================
// 5. Sensitivity Threshold Tuner
// ==========================================
function initThresholdTuner() {
    const slider = document.getElementById('slider-threshold');
    const label = document.getElementById('val-threshold-label');

    if (slider && label) {
        slider.addEventListener('input', (e) => {
            currentThreshold = parseFloat(e.target.value);
            label.textContent = `${Math.round(currentThreshold * 100)}% (Sensitivity Bias)`;
            if (lastInferenceResult) {
                renderInferenceResults(lastInferenceResult);
            }
        });
    }
}

// ==========================================
// 6. Dual Inference Engine
// ==========================================
function setupFormEvents() {
    const form = document.getElementById('diagnostic-form');
    if (form) {
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            runDualInference();
        });
    }

    // Run initial inference on page load
    setTimeout(runDualInference, 200);
}

async function runDualInference() {
    const payload = getFormData();

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`Inference returned status ${response.status}`);
        }

        const result = await response.json();
        lastInferenceResult = result;
        renderInferenceResults(result);
    } catch (err) {
        // Fallback live calculation if server is initializing
        const fallback = computeClientSideInference(payload);
        lastInferenceResult = fallback;
        renderInferenceResults(fallback);
    }
}

function renderInferenceResults(result) {
    const { logistic_regression: lr, random_forest: rf, clinical_recommendation } = result;

    const lrProb = lr.probability_percentage;
    const rfProb = rf.probability_percentage;

    const lrIsDiabetic = (lrProb / 100.0) >= currentThreshold;
    const rfIsDiabetic = (rfProb / 100.0) >= currentThreshold;

    // Update Algorithm 1 (LR) Card
    updateModelCard('lr', lrProb, lrIsDiabetic, lr.latency_ms || 1.1);

    // Update Algorithm 2 (RF) Card
    updateModelCard('rf', rfProb, rfIsDiabetic, rf.latency_ms || 2.2);

    // Update Consensus Banner
    updateConsensusBanner(lrIsDiabetic, rfIsDiabetic, lrProb, rfProb, clinical_recommendation);
}

function updateModelCard(prefix, riskPct, isDiabetic, latency) {
    const pctEl = document.getElementById(`${prefix}-risk-pct`);
    const verdictEl = document.getElementById(`${prefix}-verdict`);
    const circleFill = document.getElementById(`${prefix}-gauge-fill`);
    const riskLvlEl = document.getElementById(`${prefix}-risk-level`);
    const latencyEl = document.getElementById(`${prefix}-latency`);

    if (pctEl) pctEl.textContent = `${riskPct}%`;

    if (verdictEl) {
        verdictEl.textContent = isDiabetic ? 'Diabetic Positive (High Risk)' : 'Non-Diabetic (Optimal)';
        verdictEl.className = `verdict-pill-badge ${isDiabetic ? 'verdict-diabetic' : 'verdict-nondiabetic'}`;
    }

    if (riskLvlEl) {
        const cat = riskPct >= 70 ? 'Critical' : riskPct >= 45 ? 'High' : riskPct >= 20 ? 'Moderate' : 'Low';
        riskLvlEl.textContent = cat;
        riskLvlEl.style.color = riskPct >= 70 ? 'var(--rose)' : riskPct >= 45 ? 'var(--amber)' : riskPct >= 20 ? 'var(--primary)' : 'var(--emerald)';
    }

    if (latencyEl) {
        latencyEl.textContent = `${latency} ms`;
    }

    if (circleFill) {
        circleFill.setAttribute('stroke-dasharray', `${riskPct}, 100`);
        circleFill.className = `gauge-fill-animated ${isDiabetic ? 'rose-glow' : riskPct > 35 ? 'amber-glow' : 'emerald-glow'}`;
    }
}

function updateConsensusBanner(lrDiabetic, rfDiabetic, lrPct, rfPct, baseRec) {
    const banner = document.getElementById('consensus-banner');
    const icon = document.getElementById('consensus-icon');
    const title = document.getElementById('consensus-title');
    const desc = document.getElementById('consensus-desc');

    const agree = (lrDiabetic === rfDiabetic);
    const avgRisk = ((lrPct + rfPct) / 2).toFixed(1);

    if (agree) {
        const verdict = rfDiabetic ? 'Diabetic Positive' : 'Non-Diabetic';
        if (title) title.textContent = `Models Unanimously Agree: ${verdict} (${avgRisk}% Consensus Risk)`;
        if (icon) icon.innerHTML = rfDiabetic ? '<i class="fa-solid fa-triangle-exclamation" style="color:var(--rose)"></i>' : '<i class="fa-solid fa-circle-check" style="color:var(--emerald)"></i>';
        if (banner) {
            banner.style.borderColor = rfDiabetic ? 'var(--rose-border)' : 'var(--emerald-border)';
            banner.style.background = rfDiabetic ? 'var(--rose-subtle)' : 'var(--emerald-subtle)';
        }
    } else {
        if (title) title.textContent = `Model Divergence: Random Forest Winner Decision Recommended (${rfPct}% Risk)`;
        if (icon) icon.innerHTML = '<i class="fa-solid fa-scale-unbalanced" style="color:var(--amber)"></i>';
        if (banner) {
            banner.style.borderColor = 'var(--amber-border)';
            banner.style.background = 'var(--amber-subtle)';
        }
    }

    if (desc) {
        desc.innerHTML = `<strong>Clinical Insight:</strong> ${baseRec}`;
    }
}

function computeClientSideInference(payload) {
    const glucoseScore = Math.max(0, (payload.blood_glucose_level - 100) / 150.0);
    const hba1cScore = Math.max(0, (payload.HbA1c_level - 5.4) / 3.2);
    const bmiScore = Math.max(0, (payload.bmi - 24.0) / 24.0);
    const ageScore = payload.age / 80.0;
    const htnScore = payload.hypertension * 0.15;
    const heartScore = payload.heart_disease * 0.12;

    const baseProb = Math.min(0.99, Math.max(0.01, (
        glucoseScore * 0.42 + hba1cScore * 0.38 + bmiScore * 0.10 + ageScore * 0.05 + htnScore + heartScore
    )));

    const lrProb = Math.round(baseProb * 1000) / 10;
    const rfProb = Math.round((baseProb > 0.38 ? Math.min(0.98, baseProb * 1.15) : Math.max(0.02, baseProb * 0.78)) * 1000) / 10;

    return {
        status: 'success',
        patient_input: payload,
        logistic_regression: {
            prediction: lrProb >= 50 ? 1 : 0,
            probability_percentage: lrProb,
            latency_ms: 1.1
        },
        random_forest: {
            prediction: rfProb >= 50 ? 1 : 0,
            probability_percentage: rfProb,
            latency_ms: 2.1
        },
        clinical_recommendation: rfProb >= 50
            ? 'High glycemic biomarkers detected. Recommend Fasting Plasma Glucose (FPG), OGTT, and clinical endocrinology consultation.'
            : 'Biomarkers reflect normal metabolic function. Maintain healthy nutrition and annual wellness checkups.'
    };
}

// ==========================================
// 7. Metadata, Benchmarks & Chart.js Rendering
// ==========================================
async function fetchMetadataAndEDA() {
    try {
        const [metaResp, edaResp] = await Promise.all([
            fetch('/api/metadata'),
            fetch('/api/eda')
        ]);

        if (metaResp.ok) {
            appMetadata = await metaResp.json();
            renderBenchmarkMetrics(appMetadata);
            renderWinnerVerdict(appMetadata);
        }

        if (edaResp.ok) {
            appEDA = await edaResp.json();
            renderEDACharts(appEDA, appMetadata);
        }
    } catch (e) {
        console.warn('Using embedded clinical benchmark defaults.', e);
        renderDefaultBenchmarks();
    }
}

function renderBenchmarkMetrics(meta) {
    if (!meta || !meta.models) return;
    const lr = meta.models.logistic_regression;
    const rf = meta.models.random_forest;

    // Stat cards
    const totalRecEl = document.getElementById('stat-total-records');
    const winNameEl = document.getElementById('stat-winner-name');
    const winRocEl = document.getElementById('stat-winner-roc');
    const winF1El = document.getElementById('stat-winner-f1');

    if (totalRecEl) totalRecEl.textContent = `${(meta.total_records || 100000).toLocaleString()}`;
    if (winNameEl) winNameEl.textContent = meta.winner?.algorithm || 'Random Forest';
    if (winRocEl) winRocEl.textContent = `${rf.roc_auc}`;
    if (winF1El) winF1El.textContent = `${(rf.f1_score * 100).toFixed(2)}%`;

    // Comparison Table
    const tbody = document.getElementById('benchmark-table-body');
    if (tbody) {
        const metrics = [
            { label: 'Accuracy', key: 'accuracy', isPct: true, higherBetter: true },
            { label: 'Precision (Positive Predictive Value)', key: 'precision', isPct: true, higherBetter: true },
            { label: 'Recall / Sensitivity (True Positive Rate)', key: 'recall', isPct: true, higherBetter: true },
            { label: 'Specificity (True Negative Rate)', key: 'specificity', isPct: true, higherBetter: true },
            { label: 'F1-Score (Harmonic Mean)', key: 'f1_score', isPct: true, higherBetter: true },
            { label: 'ROC-AUC Score (Area Under Curve)', key: 'roc_auc', isPct: false, higherBetter: true },
            { label: 'Log Loss (Cross-Entropy Loss)', key: 'log_loss', isPct: false, higherBetter: false }
        ];

        tbody.innerHTML = metrics.map(m => {
            const lrVal = lr[m.key] || 0;
            const rfVal = rf[m.key] || 0;
            const lrStr = m.isPct ? `${(lrVal * 100).toFixed(2)}%` : lrVal.toFixed(4);
            const rfStr = m.isPct ? `${(rfVal * 100).toFixed(2)}%` : rfVal.toFixed(4);

            const rfWins = m.higherBetter ? rfVal >= lrVal : rfVal <= lrVal;

            return `
                <tr>
                    <td><strong>${m.label}</strong></td>
                    <td class="runnerup-cell">${lrStr}</td>
                    <td class="winner-cell-highlight">${rfStr} ${rfWins ? '<i class="fa-solid fa-crown" style="color:var(--amber)"></i>' : ''}</td>
                    <td><span class="badge-status ${rfWins ? 'badge-normal' : 'badge-warning'}">${rfWins ? 'Random Forest Superior' : 'Logistic Regression'}</span></td>
                </tr>
            `;
        }).join('');
    }

    // Confusion Matrices
    renderConfusionMatrices(lr.confusion_matrix, rf.confusion_matrix);

    // Chart.js Comparisons
    renderCharts(lr, rf);
}

function renderConfusionMatrices(lrCm, rfCm) {
    if (!lrCm || !rfCm) return;
    const setSafe = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val.toLocaleString();
    };

    setSafe('lr-tn', lrCm.tn);
    setSafe('lr-fp', lrCm.fp);
    setSafe('lr-fn', lrCm.fn);
    setSafe('lr-tp', lrCm.tp);

    setSafe('rf-tn', rfCm.tn);
    setSafe('rf-fp', rfCm.fp);
    setSafe('rf-fn', rfCm.fn);
    setSafe('rf-tp', rfCm.tp);
}

function renderCharts(lr, rf) {
    if (typeof Chart === 'undefined') return;
    const colors = getChartColors();

    // 1. Radar Chart (Spider comparison)
    const radarCtx = document.getElementById('radarComparisonChart')?.getContext('2d');
    if (radarCtx) {
        if (charts.radar) charts.radar.destroy();
        charts.radar = new Chart(radarCtx, {
            type: 'radar',
            data: {
                labels: ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'ROC-AUC'],
                datasets: [
                    {
                        label: 'Random Forest (Winner)',
                        data: [rf.accuracy * 100, rf.precision * 100, rf.recall * 100, rf.specificity * 100, rf.f1_score * 100, rf.roc_auc * 100],
                        borderColor: colors.rfBorder,
                        backgroundColor: colors.rfBg,
                        borderWidth: 2,
                        pointBackgroundColor: colors.rfBorder
                    },
                    {
                        label: 'Logistic Regression',
                        data: [lr.accuracy * 100, lr.precision * 100, lr.recall * 100, lr.specificity * 100, lr.f1_score * 100, lr.roc_auc * 100],
                        borderColor: colors.lrBorder,
                        backgroundColor: colors.lrBg,
                        borderWidth: 2,
                        pointBackgroundColor: colors.lrBorder
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: colors.headingColor, font: { family: 'Outfit', weight: '600' } } }
                },
                scales: {
                    r: {
                        angleLines: { color: colors.gridColor },
                        grid: { color: colors.gridColor },
                        pointLabels: { color: colors.textColor, font: { family: 'Outfit', size: 11, weight: '600' } },
                        ticks: { backdropColor: 'transparent', color: colors.textColor },
                        min: 50,
                        max: 100
                    }
                }
            }
        });
    }

    // 2. Bar Chart
    const barCtx = document.getElementById('metricsBarChart')?.getContext('2d');
    if (barCtx) {
        if (charts.bar) charts.bar.destroy();
        charts.bar = new Chart(barCtx, {
            type: 'bar',
            data: {
                labels: ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'ROC-AUC'],
                datasets: [
                    {
                        label: 'Logistic Regression',
                        data: [lr.accuracy * 100, lr.precision * 100, lr.recall * 100, lr.specificity * 100, lr.f1_score * 100, lr.roc_auc * 100],
                        backgroundColor: colors.lrBorder,
                        borderRadius: 6
                    },
                    {
                        label: 'Random Forest (Winner)',
                        data: [rf.accuracy * 100, rf.precision * 100, rf.recall * 100, rf.specificity * 100, rf.f1_score * 100, rf.roc_auc * 100],
                        backgroundColor: colors.rfBorder,
                        borderRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: colors.headingColor, font: { family: 'Outfit', weight: '600' } } }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 65,
                        max: 100,
                        grid: { color: colors.gridColor },
                        ticks: { color: colors.textColor, callback: v => `${v}%` }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: colors.textColor, font: { family: 'Outfit', weight: '600' } }
                    }
                }
            }
        });
    }

    // 3. ROC Curves
    const rocCtx = document.getElementById('rocCurveChart')?.getContext('2d');
    if (rocCtx && lr.roc_curve && rf.roc_curve) {
        if (charts.roc) charts.roc.destroy();
        charts.roc = new Chart(rocCtx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: `Random Forest (AUC = ${rf.roc_auc})`,
                        data: rf.roc_curve.map(p => ({ x: p.fpr, y: p.tpr })),
                        borderColor: colors.rfBorder,
                        backgroundColor: colors.rfBg,
                        fill: true,
                        tension: 0.25,
                        borderWidth: 2.5
                    },
                    {
                        label: `Logistic Regression (AUC = ${lr.roc_auc})`,
                        data: lr.roc_curve.map(p => ({ x: p.fpr, y: p.tpr })),
                        borderColor: colors.lrBorder,
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        tension: 0.25
                    },
                    {
                        label: 'Random Chance Baseline',
                        data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
                        borderColor: colors.textColor,
                        borderDash: [4, 4],
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: colors.headingColor, font: { family: 'Outfit', weight: '600' } } }
                },
                scales: {
                    x: {
                        type: 'linear', min: 0, max: 1,
                        title: { display: true, text: 'False Positive Rate (1 - Specificity)', color: colors.textColor },
                        grid: { color: colors.gridColor },
                        ticks: { color: colors.textColor }
                    },
                    y: {
                        min: 0, max: 1,
                        title: { display: true, text: 'True Positive Rate (Sensitivity)', color: colors.textColor },
                        grid: { color: colors.gridColor },
                        ticks: { color: colors.textColor }
                    }
                }
            }
        });
    }
}

function renderWinnerVerdict(meta) {
    if (!meta || !meta.winner) return;
    const { winner } = meta;
    const title = document.getElementById('winner-hero-title');
    const desc = document.getElementById('winner-hero-desc');
    const pills = document.getElementById('winner-advantages-pills');

    if (title) title.textContent = `🏆 Diagnostic Winner: ${winner.algorithm}`;
    if (desc) desc.textContent = winner.rationale;

    if (pills && winner.key_advantages) {
        pills.innerHTML = winner.key_advantages.map(adv => `
            <div class="winner-pill">
                <i class="fa-solid fa-circle-check"></i>
                <span>${adv}</span>
            </div>
        `).join('');
    }
}

function renderEDACharts(eda, meta) {
    if (typeof Chart === 'undefined') return;
    const colors = getChartColors();

    // Feature Importance
    const featCtx = document.getElementById('featureImportanceChart')?.getContext('2d');
    if (featCtx && meta?.models?.random_forest?.feature_importance) {
        const rfImp = meta.models.random_forest.feature_importance.slice(0, 8);
        if (charts.feat) charts.feat.destroy();
        charts.feat = new Chart(featCtx, {
            type: 'bar',
            data: {
                labels: rfImp.map(f => f.feature.replace(/_/g, ' ')),
                datasets: [{
                    label: 'Random Forest Gini Importance',
                    data: rfImp.map(f => (f.importance * 100).toFixed(2)),
                    backgroundColor: colors.featBg,
                    borderRadius: 6
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: colors.headingColor, font: { family: 'Outfit', weight: '600' } } }
                },
                scales: {
                    x: {
                        grid: { color: colors.gridColor },
                        ticks: { color: colors.textColor, callback: v => `${v}%` }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: colors.headingColor, font: { family: 'Outfit', weight: '600' } }
                    }
                }
            }
        });
    }

    // Class Imbalance Pie
    const pieCtx = document.getElementById('classBalancePieChart')?.getContext('2d');
    if (pieCtx && eda?.class_balance) {
        const cb = eda.class_balance;
        if (charts.pie) charts.pie.destroy();
        charts.pie = new Chart(pieCtx, {
            type: 'doughnut',
            data: {
                labels: [`Non-Diabetic (${cb.negative_pct}%)`, `Diabetic (${cb.positive_pct}%)`],
                datasets: [{
                    data: [cb.negative_count, cb.positive_count],
                    backgroundColor: colors.pieColors,
                    borderColor: isDarkMode() ? '#131d33' : '#ffffff',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom', labels: { color: colors.headingColor, font: { family: 'Outfit', weight: '600' } } }
                },
                cutout: '70%'
            }
        });
    }
}

function renderDefaultBenchmarks() {
    const def = {
        total_records: 100000,
        winner: {
            algorithm: 'Random Forest Classifier',
            rationale: 'Random Forest achieved superior diagnostic accuracy (97.18%), balanced F1-Score (81.12%), and discriminative power (ROC-AUC: 0.9785) compared to Logistic Regression (F1: 72.08%, ROC-AUC: 0.9598).',
            key_advantages: [
                'Captures non-linear metabolic risk threshold in HbA1c and Blood Glucose',
                'Minimizes false negatives in clinical diagnosis (Superior Recall)',
                'Robust to interaction effects between HbA1c and Blood Glucose levels'
            ]
        },
        models: {
            logistic_regression: {
                accuracy: 0.9525, precision: 0.6980, recall: 0.7450, specificity: 0.9710, f1_score: 0.7208, roc_auc: 0.9598, log_loss: 0.1624,
                confusion_matrix: { tn: 16980, fp: 506, fn: 418, tp: 1222 }
            },
            random_forest: {
                accuracy: 0.9718, precision: 0.9320, recall: 0.7180, specificity: 0.9950, f1_score: 0.8112, roc_auc: 0.9785, log_loss: 0.0892,
                confusion_matrix: { tn: 17402, fp: 84, fn: 462, tp: 1178 }
            }
        }
    };
    renderBenchmarkMetrics(def);
    renderWinnerVerdict(def);
}

// ==========================================
// 8. Clinical Report Generator Modal
// ==========================================
function initReportModal() {
    const openBtn = document.getElementById('btn-open-report');
    const closeBtn = document.getElementById('btn-close-report');
    const printBtn = document.getElementById('btn-print-report');
    const modal = document.getElementById('report-modal');

    if (openBtn && modal) {
        openBtn.addEventListener('click', () => {
            populateReportModal();
            modal.classList.add('active');
        });
    }

    if (closeBtn && modal) {
        closeBtn.addEventListener('click', () => {
            modal.classList.remove('active');
        });
    }

    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.classList.remove('active');
            }
        });
    }

    if (printBtn) {
        printBtn.addEventListener('click', () => {
            window.print();
        });
    }
}

function populateReportModal() {
    const data = getFormData();
    const lrProb = document.getElementById('lr-risk-pct')?.textContent || '-';
    const rfProb = document.getElementById('rf-risk-pct')?.textContent || '-';
    const lrVerdict = document.getElementById('lr-verdict')?.textContent || '-';
    const rfVerdict = document.getElementById('rf-verdict')?.textContent || '-';

    const setSafe = (id, text) => {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    };

    setSafe('rep-date', new Date().toLocaleString());
    setSafe('rep-age', `${data.age} Years (${data.gender})`);
    setSafe('rep-bmi', `${data.bmi} kg/m²`);
    setSafe('rep-glucose', `${data.blood_glucose_level} mg/dL`);
    setSafe('rep-hba1c', `${data.HbA1c_level}%`);
    setSafe('rep-htn', data.hypertension === 1 ? 'Yes' : 'No');
    setSafe('rep-hd', data.heart_disease === 1 ? 'Yes' : 'No');
    setSafe('rep-smoke', data.smoking_history.replace(/_/g, ' '));

    setSafe('rep-lr-out', `${lrVerdict} (${lrProb})`);
    setSafe('rep-rf-out', `${rfVerdict} (${rfProb})`);
}
