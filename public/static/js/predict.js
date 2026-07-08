// public/static/js/predict.js
// Handles prediction form, result display, charting and simple advice.

(function () {
  // Helper to dynamically load an external script (Chart.js)
  function loadScript(src, cb) {
    if (document.querySelector('script[src="' + src + '"]')) {
      return cb && cb();
    }
    var s = document.createElement('script');
    s.src = src;
    s.onload = function () { cb && cb(); };
    s.onerror = function () { cb && cb(new Error('Failed to load ' + src)); };
    document.head.appendChild(s);
  }

  // Small DOM helpers
  var form = document.getElementById('predictionForm');
  var runBtn = document.getElementById('runPredictBtn');
  var spinner = document.getElementById('predictSpinner');
  var resultCard = document.getElementById('predictionResult');
  var errorBox = document.getElementById('predictionError');

  var valueEl = document.getElementById('predictionValue');
  var metaMethod = document.getElementById('metaMethod');
  var metaModel = document.getElementById('metaModel');
  var metaTrainMAE = document.getElementById('metaTrainMAE');
  var metaTestMAE = document.getElementById('metaTestMAE');
  var metaTimeSteps = document.getElementById('metaTimeSteps');
  var adviceEl = document.getElementById('predictionAdvice');

  // Chart variables
  var chartCanvasId = 'predictionChart';
  var chartInstance = null;
  var chartLoaded = false;

  function showSpinner(show) {
    spinner.style.display = show ? 'inline' : 'none';
    runBtn.disabled = show;
  }

  function showError(msg) {
    errorBox.hidden = false;
    errorBox.textContent = msg;
  }

  function clearError() {
    errorBox.hidden = true;
    errorBox.textContent = '';
  }

  function formatNumber(n) {
    return (Math.round(n * 100) / 100).toFixed(2);
  }

  function renderChart(inputs, prediction) {
    // create canvas if not present
    var canvas = document.getElementById(chartCanvasId);
    if (!canvas) {
      canvas = document.createElement('canvas');
      canvas.id = chartCanvasId;
      canvas.style.maxWidth = '100%';
      // insert after resultValue
      valueEl.parentNode.insertBefore(canvas, valueEl.nextSibling);
    }

    function draw() {
      var labels = [];
      var data = [];
      // Inputs are newest-first in the form; reverse so x-axis is oldest -> latest
      for (var i = inputs.length - 1; i >= 0; i--) {
        labels.push('T-' + (inputs.length - i));
        data.push(inputs[i]);
      }
      labels.push('T+1');
      data.push(prediction);

      var ctx = canvas.getContext('2d');
      if (chartInstance) {
        chartInstance.data.labels = labels;
        chartInstance.data.datasets[0].data = data;
        chartInstance.update();
        return;
      }

      chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
          labels: labels,
          datasets: [{
            label: 'Stock level',
            data: data,
            fill: false,
            borderColor: '#2a9d8f',
            backgroundColor: '#2a9d8f',
            pointRadius: 4,
            tension: 0.25,
          }]
        },
        options: {
          responsive: true,
          scales: {
            y: {
              beginAtZero: true
            }
          },
          plugins: {
            legend: { display: false }
          }
        }
      });
    }

    if (typeof Chart === 'undefined') {
      // load Chart.js from CDN, then draw
      loadScript('https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js', function (err) {
        if (err) {
          // Chart failed to load — silently ignore chart
          console.warn('Chart.js failed to load:', err);
          return;
        }
        draw();
      });
    } else {
      draw();
    }
  }

  function computeAdvice(prediction, inputs) {
    // Simple rule-based advice:
    var latest = inputs[0] || 0;
    var avg = inputs.reduce(function (a, b) { return a + b; }, 0) / Math.max(1, inputs.length);

    if (prediction < 0) {
      return 'Predicted level is below zero — check input data or model.';
    }
    var pctDrop = ((latest - prediction) / Math.max(1, latest)) * 100;
    if (prediction <= 0.2 * avg) {
      return 'Warning: predicted level is very low compared to recent history — consider reordering immediately.';
    }
    if (pctDrop >= 30) {
      return 'Significant decline expected vs latest period — consider replenishing stock.';
    }
    if (prediction < avg) {
      return 'Slight decline predicted vs recent average — monitor closely and consider reorder thresholds.';
    }
    return 'Predicted level looks stable or increasing compared to recent history.';
  }

  form && form.addEventListener('submit', function (ev) {
    ev.preventDefault();
    clearError();
    resultCard.hidden = true;
    // collect all inputs with name starting quantity
    var inputs = [];
    var fields = form.querySelectorAll('input[name^="quantity"]');
    for (var i = 0; i < fields.length; i++) {
      var v = fields[i].value;
      if (v === '') {
        showError('Please fill all quantity fields.');
        return;
      }
      var num = parseFloat(v);
      if (isNaN(num) || num < 0) {
        showError('Quantities must be non-negative numbers.');
        return;
      }
      inputs.push(num);
    }

    // build payload: quantity1..N (fields are assumed newest-first)
    var payload = {};
    for (var j = 0; j < inputs.length; j++) {
      payload['quantity' + (j + 1)] = inputs[j];
    }

    showSpinner(true);

    fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
      .then(function (res) {
        showSpinner(false);
        if (!res.ok) {
          return res.json().then(function (body) {
            var msg = body && body.error ? body.error : 'Server error';
            throw new Error(msg);
          }).catch(function () {
            throw new Error('Server error');
          });
        }
        return res.json();
      })
      .then(function (data) {
        if (!data || data.success !== true) {
          var err = (data && data.error) ? data.error : 'Prediction failed';
          throw new Error(err);
        }

        // show model metadata and prediction
        var pred = parseFloat(data.prediction);
        valueEl.textContent = formatNumber(pred);

        metaMethod.textContent = data.method || 'unknown';
        metaModel.textContent = data.model_type || 'n/a';
        metaTrainMAE.textContent = (data.train_mae !== undefined && data.train_mae !== null) ? formatNumber(data.train_mae) : 'n/a';
        metaTestMAE.textContent = (data.test_mae !== undefined && data.test_mae !== null) ? formatNumber(data.test_mae) : 'n/a';
        metaTimeSteps.textContent = data.time_steps || inputs.length;

        // confidence interval using test_mae (fallback train_mae, fallback 10%)
        var errorMargin = null;
        if (data.test_mae) errorMargin = parseFloat(data.test_mae);
        else if (data.train_mae) errorMargin = parseFloat(data.train_mae);
        else errorMargin = Math.abs(pred) * 0.1;

        var lower = pred - errorMargin;
        var upper = pred + errorMargin;
        var ciText = 'Prediction: ' + formatNumber(pred) + ' (± ' + formatNumber(errorMargin) + ') → [' + formatNumber(lower) + ' – ' + formatNumber(upper) + ']';
        valueEl.textContent = ciText;

        // actionable advice (compares prediction to recent inputs)
        var advice = computeAdvice(pred, inputs);
        adviceEl.textContent = advice;

        // show result card
        resultCard.hidden = false;
        clearError();

        // draw chart (inputs newest-first)
        renderChart(inputs, pred);
      })
      .catch(function (err) {
        showSpinner(false);
        showError(err.message || 'Prediction error');
      });
  });

  // Train button: POST to /train and show immediate feedback
  var trainBtn = document.getElementById('trainModelBtn');
  if (trainBtn) {
    trainBtn.addEventListener('click', function () {
      clearError();
      showSpinner(true);
      fetch('/train', { method: 'POST' })
        .then(function (r) { return r.json(); })
        .then(function (d) {
          showSpinner(false);
          if (d && d.success) {
            // reload page to update modelLoaded indicator and meta
            window.location.reload();
          } else {
            showError(d.error || 'Training failed');
          }
        })
        .catch(function (err) {
          showSpinner(false);
          showError(err.message || 'Training error');
        });
    });
  }
})();
