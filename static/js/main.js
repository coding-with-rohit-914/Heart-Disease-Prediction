/**
 * HeartGuard AI — Custom JavaScript
 * Project: Heart Disease Risk Prediction
 * MCA Project | Trinity Academy of Engineering, Pune
 */

'use strict';

/* ── Loading overlay ── */
const overlay = document.getElementById('loading-overlay');
function showLoading(msg = 'Analyzing patient data...') {
  if (!overlay) return;
  overlay.querySelector('.loading-text').textContent = msg;
  overlay.classList.add('active');
}
function hideLoading() {
  if (overlay) overlay.classList.remove('active');
}

/* ── Animate numbers (counters) ── */
function animateCounter(el, target, duration = 1200) {
  let start = 0;
  const step = target / (duration / 16);
  const timer = setInterval(() => {
    start += step;
    if (start >= target) { el.textContent = target; clearInterval(timer); return; }
    el.textContent = Math.floor(start);
  }, 16);
}

/* ── Animate on scroll ── */
function initScrollAnimation() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('visible');
        observer.unobserve(e.target);
      }
    });
  }, { threshold: 0.15 });
  document.querySelectorAll('.animate-on-scroll').forEach(el => observer.observe(el));
}

/* ── Risk needle positioning ── */
function setRiskNeedle(probability) {
  const needle = document.querySelector('.risk-needle');
  if (!needle) return;
  setTimeout(() => { needle.style.left = `${probability}%`; }, 400);
}

/* ── Probability bar animation ── */
function animateProbBar(id, value) {
  const el = document.getElementById(id);
  if (!el) return;
  setTimeout(() => { el.style.width = value + '%'; }, 300);
}

/* ── Form: highlight active model radio ── */
function initModelRadios() {
  const radios = document.querySelectorAll('input[name="model"]');
  radios.forEach(r => {
    r.addEventListener('change', function () {
      document.querySelectorAll('.model-option').forEach(l => l.classList.remove('active'));
      this.closest('.model-option')?.classList.add('active');
    });
  });
  // Activate first
  const first = document.querySelector('input[name="model"]');
  if (first) first.closest('.model-option')?.classList.add('active');
}

/* ── Sample data loader ── */
const SAMPLE_DATA = {
  high: {
    age: 63, sex: 1, cp: 0, trestbps: 145, chol: 233,
    fbs: 1, restecg: 0, thalach: 150, exang: 0,
    oldpeak: 2.3, slope: 0, ca: 0, thal: 1
  },
  low: {
    age: 41, sex: 0, cp: 2, trestbps: 130, chol: 204,
    fbs: 0, restecg: 2, thalach: 172, exang: 0,
    oldpeak: 1.4, slope: 2, ca: 0, thal: 2
  },
  medium: {
    age: 52, sex: 1, cp: 1, trestbps: 135, chol: 264,
    fbs: 0, restecg: 0, thalach: 143, exang: 0,
    oldpeak: 1.5, slope: 1, ca: 0, thal: 2
  }
};

function loadSample(type) {
  const s = SAMPLE_DATA[type];
  if (!s) return;
  Object.entries(s).forEach(([key, val]) => {
    const el = document.querySelector(`[name="${key}"]`);
    if (el) el.value = val;
  });
  // Visual feedback
  showToast(`✅ ${type.charAt(0).toUpperCase() + type.slice(1)} Risk sample loaded!`, 'success');
}

/* ── Toast notification ── */
function showToast(message, type = 'info') {
  const colors = { success: '#2a9d8f', danger: '#e63946', info: '#1d3557', warning: '#f4a261' };
  const toast = document.createElement('div');
  toast.innerHTML = message;
  Object.assign(toast.style, {
    position: 'fixed', bottom: '24px', right: '24px',
    background: colors[type] || colors.info,
    color: '#fff', padding: '12px 22px',
    borderRadius: '12px', fontWeight: '600',
    fontSize: '0.9rem', zIndex: '9999',
    boxShadow: '0 4px 16px rgba(0,0,0,0.2)',
    opacity: '0', transition: 'opacity 0.3s ease',
    maxWidth: '320px'
  });
  document.body.appendChild(toast);
  requestAnimationFrame(() => { toast.style.opacity = '1'; });
  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

/* ── Predict form submit with spinner ── */
function initPredictForm() {
  const form = document.getElementById('predictForm');
  if (!form) return;
  form.addEventListener('submit', function (e) {
    // Basic validation
    const required = form.querySelectorAll('[required]');
    let valid = true;
    required.forEach(el => {
      if (!el.value) { el.classList.add('is-invalid'); valid = false; }
      else el.classList.remove('is-invalid');
    });
    if (!valid) { e.preventDefault(); showToast('⚠️ Please fill all required fields.', 'danger'); return; }

    const btn = form.querySelector('button[type="submit"]');
    if (btn) {
      btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Analyzing...';
      btn.disabled = true;
    }
    showLoading('Running ML model...');
  });
}

/* ── Dashboard: animate stat numbers ── */
function initStatCounters() {
  document.querySelectorAll('.stat-number').forEach(el => {
    const val = parseInt(el.textContent);
    if (!isNaN(val) && val > 0) {
      el.textContent = '0';
      animateCounter(el, val);
    }
  });
}

/* ── Initialize all on DOM ready ── */
document.addEventListener('DOMContentLoaded', () => {
  initScrollAnimation();
  initModelRadios();
  initPredictForm();
  initStatCounters();

  // Bootstrap tooltips
  document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => {
    new bootstrap.Tooltip(el);
  });

  // Auto-dismiss alerts after 5s
  document.querySelectorAll('.alert-dismissible').forEach(alert => {
    setTimeout(() => {
      const btn = alert.querySelector('.btn-close');
      if (btn) btn.click();
    }, 5000);
  });
});

/* ── Expose globals for inline use ── */
window.loadSample  = loadSample;
window.showToast   = showToast;
window.showLoading = showLoading;
window.hideLoading = hideLoading;
window.setRiskNeedle    = setRiskNeedle;
window.animateProbBar   = animateProbBar;
