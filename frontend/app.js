// Supply Chain Control Tower - JavaScript

const API_BASE = 'http://localhost:8000/api';

// Navigation
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
        const page = item.dataset.page;
        showPage(page);

        document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
        item.classList.add('active');
    });
});

document.querySelectorAll('.module-card').forEach(card => {
    card.addEventListener('click', () => {
        const page = card.dataset.page;
        showPage(page);

        document.querySelectorAll('.nav-item').forEach(i => {
            i.classList.toggle('active', i.dataset.page === page);
        });
    });
});

function showPage(pageId) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.getElementById(`page-${pageId}`).classList.add('active');
}

// Slider value displays
document.getElementById('delay-traffic')?.addEventListener('input', e => {
    document.getElementById('traffic-val').textContent = Math.round(e.target.value * 100) + '%';
});
document.getElementById('delay-driver')?.addEventListener('input', e => {
    document.getElementById('driver-val').textContent = Math.round(e.target.value * 100) + '%';
});
document.getElementById('delay-disruption')?.addEventListener('input', e => {
    document.getElementById('disruption-val').textContent = Math.round(e.target.value * 100) + '%';
});
document.getElementById('risk-route')?.addEventListener('input', e => {
    document.getElementById('route-val').textContent = Math.round(e.target.value * 100) + '%';
});
document.getElementById('risk-supplier')?.addEventListener('input', e => {
    document.getElementById('supplier-val').textContent = Math.round(e.target.value * 100) + '%';
});
document.getElementById('risk-disruption')?.addEventListener('input', e => {
    document.getElementById('disrupt-val').textContent = Math.round(e.target.value * 100) + '%';
});
document.getElementById('risk-delay')?.addEventListener('input', e => {
    document.getElementById('delayprob-val').textContent = Math.round(e.target.value * 100) + '%';
});
document.getElementById('forecast-horizon')?.addEventListener('input', e => {
    document.getElementById('horizon-val').textContent = e.target.value;
});
document.getElementById('anomaly-traffic')?.addEventListener('input', e => {
    document.getElementById('anom-traffic-val').textContent = Math.round(e.target.value * 100) + '%';
});
document.getElementById('anomaly-fatigue')?.addEventListener('input', e => {
    document.getElementById('fatigue-val').textContent = Math.round(e.target.value * 100) + '%';
});

// API Calls
async function predictDelay() {
    const data = {
        lead_time_days: parseFloat(document.getElementById('delay-leadtime').value),
        weather_severity: parseInt(document.getElementById('delay-weather').value),
        traffic_level: parseFloat(document.getElementById('delay-traffic').value),
        driver_score: parseFloat(document.getElementById('delay-driver').value),
        inventory_level: parseFloat(document.getElementById('delay-inventory').value),
        disruption_score: parseFloat(document.getElementById('delay-disruption').value)
    };

    let result;
    try {
        const response = await fetch(`${API_BASE}/delay/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        result = await response.json();
    } catch (e) {
        // Calculate locally if API unavailable
        const prob = data.traffic_level * 0.35 + (data.weather_severity / 3) * 0.20 +
            (1 - data.driver_score) * 0.25 + data.disruption_score * 0.15 +
            (1 - Math.min(data.inventory_level / 500, 1)) * 0.05;
        result = {
            is_delayed: prob > 0.5,
            delay_probability: prob,
            factors: {
                traffic_impact: data.traffic_level * 0.35,
                weather_impact: (data.weather_severity / 3) * 0.20,
                driver_impact: (1 - data.driver_score) * 0.25,
                disruption_impact: data.disruption_score * 0.15
            }
        };
    }

    // Update UI
    const resultCard = document.getElementById('delay-result');
    const factorsGrid = document.getElementById('delay-factors');

    resultCard.classList.remove('hidden', 'success', 'danger');
    factorsGrid.classList.remove('hidden');

    if (result.is_delayed) {
        resultCard.classList.add('danger');
        document.getElementById('delay-result-icon').textContent = '⚠️';
        document.getElementById('delay-result-title').textContent = 'High Risk of Delay!';
    } else {
        resultCard.classList.add('success');
        document.getElementById('delay-result-icon').textContent = '✅';
        document.getElementById('delay-result-title').textContent = 'On-Time Delivery Expected';
    }

    document.getElementById('delay-result-text').innerHTML =
        `Delay Probability: <strong>${(result.delay_probability * 100).toFixed(1)}%</strong>`;

    const gauge = document.getElementById('delay-gauge');
    gauge.style.width = `${result.delay_probability * 100}%`;
    gauge.style.background = result.is_delayed ? 'var(--gradient-red)' : 'var(--gradient-green)';

    document.getElementById('factor-traffic').textContent =
        (result.factors?.traffic_impact * 100 || data.traffic_level * 35).toFixed(1) + '%';
    document.getElementById('factor-weather').textContent =
        (result.factors?.weather_impact * 100 || (data.weather_severity / 3) * 20).toFixed(1) + '%';
    document.getElementById('factor-driver').textContent =
        (result.factors?.driver_impact * 100 || (1 - data.driver_score) * 25).toFixed(1) + '%';
    document.getElementById('factor-disruption').textContent =
        (result.factors?.disruption_impact * 100 || data.disruption_score * 15).toFixed(1) + '%';
}

async function classifyRisk() {
    const data = {
        route_risk: parseFloat(document.getElementById('risk-route').value),
        supplier_reliability: parseFloat(document.getElementById('risk-supplier').value),
        disruption_likelihood: parseFloat(document.getElementById('risk-disruption').value),
        delay_probability: parseFloat(document.getElementById('risk-delay').value)
    };

    let result;
    try {
        const response = await fetch(`${API_BASE}/risk/classify`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        result = await response.json();
    } catch (e) {
        const score = data.route_risk * 0.25 + (1 - data.supplier_reliability) * 0.25 +
            data.disruption_likelihood * 0.35 + data.delay_probability * 0.15;

        if (score < 0.33) {
            result = { risk_class: 'Low Risk', confidence: 1 - score };
        } else if (score < 0.66) {
            result = { risk_class: 'Moderate Risk', confidence: 0.7 };
        } else {
            result = { risk_class: 'High Risk', confidence: score };
        }
    }

    const resultCard = document.getElementById('risk-result');
    resultCard.classList.remove('hidden', 'success', 'warning', 'danger');

    if (result.risk_class === 'Low Risk') {
        resultCard.classList.add('success');
        document.getElementById('risk-result-icon').textContent = '🟢';
    } else if (result.risk_class === 'Moderate Risk') {
        resultCard.classList.add('warning');
        document.getElementById('risk-result-icon').textContent = '🟡';
    } else {
        resultCard.classList.add('danger');
        document.getElementById('risk-result-icon').textContent = '🔴';
    }

    document.getElementById('risk-result-class').textContent = result.risk_class;
    document.getElementById('risk-result-confidence').innerHTML =
        `Confidence: <strong>${(result.confidence * 100).toFixed(1)}%</strong>`;
}

async function generateForecast() {
    const horizon = parseInt(document.getElementById('forecast-horizon').value);
    const model = document.getElementById('forecast-model').value;

    let forecastData;
    try {
        const response = await fetch(`${API_BASE}/forecast/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ horizon_hours: horizon, model_type: model })
        });
        forecastData = await response.json();
    } catch (e) {
        // Generate mock data
        forecastData = {
            forecast: Array.from({ length: horizon }, (_, i) => ({
                predicted_value: 150 + Math.sin(i / 6) * 20 + Math.random() * 10
            }))
        };
    }

    updateForecastChart(forecastData.forecast);
}

async function detectAnomaly() {
    const data = {
        iot_temperature: parseFloat(document.getElementById('anomaly-temp').value),
        fuel_consumption: parseFloat(document.getElementById('anomaly-fuel').value),
        traffic_level: parseFloat(document.getElementById('anomaly-traffic').value),
        driver_fatigue: parseFloat(document.getElementById('anomaly-fatigue').value)
    };

    let result;
    try {
        const response = await fetch(`${API_BASE}/anomaly/detect`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        result = await response.json();
    } catch (e) {
        const tempDev = Math.max(0, Math.abs(data.iot_temperature - 25) / 20);
        const fuelDev = Math.max(0, Math.abs(data.fuel_consumption - 12.5) / 15);
        const trafficDev = Math.max(0, data.traffic_level - 0.7);
        const fatigueDev = Math.max(0, data.driver_fatigue - 0.3);

        const score = (tempDev + fuelDev + trafficDev + fatigueDev) / 4;
        result = {
            is_anomaly: score > 0.3,
            anomaly_score: score,
            recommendation: score > 0.3 ? 'Investigate operational metrics' : 'All metrics normal'
        };
    }

    const resultCard = document.getElementById('anomaly-result');
    resultCard.classList.remove('hidden', 'success', 'danger');

    if (result.is_anomaly) {
        resultCard.classList.add('danger');
        document.getElementById('anomaly-result-icon').textContent = '🚨';
        document.getElementById('anomaly-result-title').textContent = 'ANOMALY DETECTED!';
    } else {
        resultCard.classList.add('success');
        document.getElementById('anomaly-result-icon').textContent = '✅';
        document.getElementById('anomaly-result-title').textContent = 'Normal Operation';
    }

    document.getElementById('anomaly-result-text').innerHTML =
        `Anomaly Score: <strong>${(result.anomaly_score * 100).toFixed(1)}%</strong>`;
    document.getElementById('anomaly-recommendation').textContent = result.recommendation;
}

// Charts
let riskChart, delayTrendChart, forecastChart;

function initCharts() {
    // Risk Distribution Chart
    const riskCtx = document.getElementById('riskChart')?.getContext('2d');
    if (riskCtx) {
        riskChart = new Chart(riskCtx, {
            type: 'doughnut',
            data: {
                labels: ['Low Risk', 'Moderate Risk', 'High Risk'],
                datasets: [{
                    data: [45, 48, 7],
                    backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
                    borderWidth: 0,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#8b8b9a', padding: 20 }
                    }
                },
                cutout: '70%'
            }
        });
    }

    // Delay Trend Chart
    const delayCtx = document.getElementById('delayTrendChart')?.getContext('2d');
    if (delayCtx) {
        delayTrendChart = new Chart(delayCtx, {
            type: 'line',
            data: {
                labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                datasets: [{
                    label: 'Avg Delay (hours)',
                    data: [2.1, 2.4, 1.9, 2.8, 2.2, 1.5, 2.3],
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 4,
                    pointBackgroundColor: '#667eea'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#8b8b9a' }
                    },
                    y: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#8b8b9a' }
                    }
                }
            }
        });
    }

    // Forecast Chart
    const forecastCtx = document.getElementById('forecastChart')?.getContext('2d');
    if (forecastCtx) {
        forecastChart = new Chart(forecastCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Historical',
                        data: [],
                        borderColor: '#4f8cff',
                        backgroundColor: 'rgba(79, 140, 255, 0.1)',
                        fill: true,
                        tension: 0.3
                    },
                    {
                        label: 'Forecast',
                        data: [],
                        borderColor: '#8b5cf6',
                        borderDash: [5, 5],
                        backgroundColor: 'rgba(139, 92, 246, 0.1)',
                        fill: true,
                        tension: 0.3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#8b8b9a', maxTicksLimit: 8 }
                    },
                    y: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#8b8b9a' }
                    }
                }
            }
        });

        // Initial data
        generateForecast();
    }
}

function updateForecastChart(forecast) {
    if (!forecastChart) return;

    const labels = forecast.map((_, i) => `+${i}h`);
    const values = forecast.map(f => f.predicted_value);

    // Simulate historical data
    const historical = Array.from({ length: 24 }, () => 140 + Math.random() * 30);
    const histLabels = Array.from({ length: 24 }, (_, i) => `-${24 - i}h`);

    forecastChart.data.labels = [...histLabels, ...labels];
    forecastChart.data.datasets[0].data = [...historical, ...Array(labels.length).fill(null)];
    forecastChart.data.datasets[1].data = [...Array(24).fill(null), ...values];
    forecastChart.update();
}

// Load alerts
async function loadAlerts() {
    const alertList = document.getElementById('alert-list');
    if (!alertList) return;

    let alerts;
    try {
        const response = await fetch(`${API_BASE}/anomaly/alerts`);
        const data = await response.json();
        alerts = data.alerts || [];
    } catch (e) {
        alerts = [
            { timestamp: '2024-08-23 21:00', anomaly_score: 0.95, iot_temperature: 42.3 },
            { timestamp: '2024-08-23 18:30', anomaly_score: 0.92, iot_temperature: 38.7 },
            { timestamp: '2024-08-22 14:15', anomaly_score: 0.91, iot_temperature: 41.2 }
        ];
    }

    document.getElementById('alert-count').textContent = alerts.length;

    alertList.innerHTML = alerts.slice(0, 5).map(alert => `
        <div class="alert-item">
            <span class="alert-time">${alert.timestamp}</span>
            <span class="alert-score">${(alert.anomaly_score * 100).toFixed(1)}%</span>
            <span class="alert-metric">${alert.iot_temperature?.toFixed(1)}°C</span>
        </div>
    `).join('');
}

// Load dashboard data
async function loadDashboardData() {
    try {
        const [delayStats, riskDist] = await Promise.all([
            fetch(`${API_BASE}/delay/stats`).then(r => r.json()).catch(() => null),
            fetch(`${API_BASE}/risk/distribution`).then(r => r.json()).catch(() => null)
        ]);

        if (delayStats) {
            document.getElementById('kpi-shipments').textContent = delayStats.total_shipments?.toLocaleString() || '32,065';
            document.getElementById('kpi-ontime').textContent = ((delayStats.on_time_rate || 0.847) * 100).toFixed(1) + '%';
            document.getElementById('kpi-delay').textContent = (delayStats.avg_delay_deviation || 2.3).toFixed(1) + 'h';
        }

        if (riskDist) {
            document.getElementById('kpi-highrisk').textContent = riskDist.high_risk_count?.toLocaleString() || '2,341';
            document.getElementById('risk-high').textContent = riskDist.high_risk_count?.toLocaleString() || '2,341';
            document.getElementById('risk-total').textContent = riskDist.total?.toLocaleString() || '32,065';

            // Update chart
            if (riskChart && riskDist.distribution) {
                const dist = riskDist.distribution;
                riskChart.data.datasets[0].data = [
                    dist['Low Risk'] || 14490,
                    dist['Moderate Risk'] || 15234,
                    dist['High Risk'] || 2341
                ];
                riskChart.update();
            }
        }
    } catch (e) {
        console.log('Using default dashboard data');
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    loadDashboardData();
    loadAlerts();
});
