// ── Toast system ──────────────────────────────────────────────
(function() {
  window.showToast = function(message, type, duration) {
    type = type || 'info';
    duration = duration || 4000;
    const container = document.getElementById('toast-container');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = 'toast' + (type === 'error' ? ' toast-error' : type === 'success' ? ' toast-success' : '');
    const icons = { error: '⚠', success: '✓', info: 'ℹ' };
    toast.innerHTML = '<span style="font-size:1rem">' + (icons[type] || 'ℹ') + '</span><span>' + message + '</span>';
    container.appendChild(toast);
    const t = setTimeout(function() { dismiss(toast); }, duration);
    toast.addEventListener('click', function() { clearTimeout(t); dismiss(toast); });
    function dismiss(el) {
      el.classList.add('dismissing');
      el.addEventListener('animationend', function() { el.remove(); }, { once: true });
    }
  };
})();

// ── Debounce ──────────────────────────────────────────────────
window.debounce = function(fn, delay) {
  var timer;
  return function() {
    var args = arguments;
    var ctx = this;
    clearTimeout(timer);
    timer = setTimeout(function() { fn.apply(ctx, args); }, delay);
  };
};

// ── Relative time for "last run" label ────────────────────────
(function() {
  var lastRunAt = null;
  window.markLastRun = function() {
    lastRunAt = Date.now();
    var lbl = document.getElementById('last-run-label');
    if (lbl) lbl.style.display = '';
    updateDisplay();
  };
  function updateDisplay() {
    if (!lastRunAt) return;
    var s = Math.round((Date.now() - lastRunAt) / 1000);
    var txt = s < 5 ? 'just now' : s < 60 ? s + 's ago' : Math.floor(s / 60) + 'm ago';
    var el = document.getElementById('last-run-time');
    if (el) el.textContent = txt;
  }
  setInterval(updateDisplay, 5000);
})();

// ── Copy to clipboard ─────────────────────────────────────────
window.copyResults = function(text) {
  navigator.clipboard.writeText(text)
    .then(function() { showToast('Copied to clipboard', 'success', 2500); })
    .catch(function() { showToast('Copy failed', 'error'); });
};

// ── Reset-to-defaults ─────────────────────────────────────────
window.registerDefaults = function(defaults) {
  var btn = document.getElementById('reset-btn');
  if (!btn) return;
  btn.addEventListener('click', function() {
    Object.keys(defaults).forEach(function(id) {
      var el = document.getElementById(id);
      if (!el) return;
      el.value = defaults[id];
    });
    showToast('Parameters reset to defaults', 'info', 2000);
    if (typeof runModule === 'function') runModule();
  });
};

// ── Collapsible sections ──────────────────────────────────────
document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.collapsible-trigger').forEach(function(trigger) {
    trigger.addEventListener('click', function() {
      var id = trigger.getAttribute('data-target');
      var content = document.getElementById(id);
      var open = trigger.getAttribute('aria-expanded') === 'true';
      trigger.setAttribute('aria-expanded', String(!open));
      content.classList.toggle('expanded', !open);
      content.classList.toggle('collapsed', open);
    });
  });
});

// ── Ctrl/Cmd+Enter keyboard shortcut ─────────────────────────
document.addEventListener('keydown', function(e) {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    if (typeof runModule === 'function') runModule();
  }
});
