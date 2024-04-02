$(document).ready(function () {
  $("#accordion .card-header button").click(function () {
    $(this).find("i").toggleClass("fa-chevron-down fa-chevron-up");
  });
});

$(document).ready(function () {
  $('[data-toggle="tooltip"]').tooltip({
    placement: "right",
  });
});

document.addEventListener("DOMContentLoaded", function () {
  document
    .getElementById("trend_method")
    .addEventListener("change", function () {
      const selectedMethod = this.value;
      const parametersContainer = document.getElementById("method_parameters");
      parametersContainer.innerHTML = "";

      if (selectedMethod === "Local Extrema") {
        parametersContainer.innerHTML = `
                <div class="parameter-indent">
                    <label for="order_days">Order Days:</label>
                    <input type="number" id="order_days" name="order_days" value="20" required>
                </div>
            `;
      } else if (selectedMethod === "MA") {
        parametersContainer.innerHTML = `
                <div class="parameter-indent">
                    <label for="ma_days">MA Days:</label>
                    <input type="number" id="ma_days" name="ma_days" placeholder="Enter MA days" value="20" required>
                    <label for="trend_days">MA trend days:</label>
                    <input type="number" id="trend_days" name="trend_days" placeholder="Enter MA trend days" value="5" required>
                </div>
            `;
      }
    });

  document.getElementById("trend_method").dispatchEvent(new Event("change"));
});

document.addEventListener("DOMContentLoaded", function () {
  const featuresContainer = document.getElementById("features_selection");
  const features = [
    {
      type: "Trend",
      method: "Local Extrema",
      oder_days: 20,
    },
    {
      type: "MACD",
      parameters: { fastperiod: 5, slowperiod: 10, signalperiod: 9 },
    },
    {
      type: "ROC",
      parameters: { trend_days: 5 },
    },
    {
      type: "Stochastic Oscillator",
      parameters: { trend_days: 5 },
    },
    {
      type: "CCI",
      parameters: { timeperiod: 14 },
    },
    {
      type: "RSI",
      parameters: { timeperiod: 14 },
    },
    {
      type: "VMA",
      parameters: { timeperiod: 20 },
    },
    {
      type: "pctChange",
      parameters: {}, // Assuming no parameters are needed
    },
    {
      type: "13W Treasury Yield",
    },
    {
      type: "5Y Treasury Yield",
    },
    {
      type: "10Y Treasury Yield",
    },
    {
      type: "30Y Treasury Yield",
    },
    {
      type: "Bollinger Bands",
      parameters: { timeperiod: 20, nbdevup: 2, nbdevdn: 2 },
    },
    {
      type: "ATR",
      parameters: { timeperiod: 14 },
    },
    {
      type: "OBV",
      parameters: {}, // Assuming no parameters are needed
    },
    {
      type: "Parabolic SAR",
      parameters: { start: 0.02, increment: 0.02, maximum: 0.2 },
    },
    {
      type: "MOM",
      parameters: { timeperiod: 10 },
    },
    {
      type: "Williams",
      parameters: { lookback_period: 14 },
    },
    {
      type: "Chaikin MF",
      parameters: { timeperiod: 20 },
    },
    // Continue adding other features as needed...
  ];

  // Dynamically create feature checkboxes and parameter inputs
  features.forEach((feature) => {
    const featureId = `feature_${feature.type
      .toLowerCase()
      .replace(/ /g, "_")}`;
    const featureDiv = document.createElement("div");
    featureDiv.classList.add("feature");
    featureDiv.innerHTML = `
      <input type="checkbox" id="${featureId}" name="${featureId}" value="${feature.type}" checked>
      <label for="${featureId}">${feature.type}</label>
      <div class="feature_params" id="params_${featureId}" style="margin-left: 20px;"></div>
    `;
    featuresContainer.appendChild(featureDiv);

    const checkbox = document.getElementById(featureId);
    checkbox.addEventListener("change", function () {
      const paramsDiv = document.getElementById(`params_${featureId}`);
      if (this.checked) {
        if (feature.type === "Trend") {
          // Special handling for Trend feature
          paramsDiv.innerHTML = `
            <label for="trend_option_${featureId}">Trend Method:</label>
            <select name="trend_option_${featureId}" id="trend_option_${featureId}">
              <option value="local_extrema">Local Extrema</option>
              <option value="ma">MA</option>
            </select>
            <div id="trend_method_params_${featureId}"></div>
          `;

          document
            .getElementById(`trend_option_${featureId}`)
            .addEventListener("change", function () {
              const methodParamsContainer = document.getElementById(
                `trend_method_params_${featureId}`
              );
              methodParamsContainer.innerHTML =
                this.value === "local_extrema"
                  ? `<label>Order Days: <input type="number" name="trend_local_extrema_order_days_${featureId}" value="20"></label>`
                  : `
                <label>MA Days: <input type="number" name="trend_ma_days_${featureId}" value="20"></label>
                <label>Trend Days: <input type="number" name="trend_trend_days_${featureId}" value="5"></label>
              `;
            });

          // Trigger the change event to initialize the parameters
          document
            .getElementById(`trend_option_${featureId}`)
            .dispatchEvent(new Event("change"));
        } else if (feature.parameters) {
          // General handling for other features with parameters
          let paramsHTML = "";
          Object.entries(feature.parameters).forEach(([param, value]) => {
            const inputType = param.toLowerCase().includes("date")
              ? "date"
              : "number";
            paramsHTML += `<label>${param}: <input type="${inputType}" name="${featureId}_${param}" value="${value}"></label><br>`;
          });
          paramsDiv.innerHTML = paramsHTML;
        }
      } else {
        paramsDiv.innerHTML = ""; // Clear parameters if unchecked
      }
    });

    // Manually trigger the change event to initialize the display
    checkbox.dispatchEvent(new Event("change"));
  });
});

function updateModelConfigFields() {
  const modelTypeSelect = document.getElementById("model_type");
  const container = document.getElementById("model_config_container");

  if (modelTypeSelect && container) {
    const modelType = modelTypeSelect.value;
    container.innerHTML = "";

    switch (modelType) {
      case "CNN_LSTM":
        container.innerHTML = `
          <div class="parameter-indent">
            <label for="conv_1_in_channels">Conv 1 In Channels:</label>
            <input type="number" id="conv_1_in_channels" name="conv_1_in_channels" value="19" required>
    
            <label for="conv_1_out_channels">Conv 1 Out Channels:</label>
            <input type="number" id="conv_1_out_channels" name="conv_1_out_channels" value="6" required>
    
            <label for="conv_1_kernel">Conv 1 Kernel Size:</label>
            <input type="number" id="conv_1_kernel" name="conv_1_kernel" value="3" required>
    
            <label for="conv_1_padding">Conv 1 Padding:</label>
            <input type="number" id="conv_1_padding" name="conv_1_padding" value="1" required>
    
            <label for="MaxPool2d_1_kernel_size">MaxPool 1 Kernel Size:</label>
            <input type="number" id="MaxPool2d_1_kernel_size" name="MaxPool2d_1_kernel_size" value="2" required>
    
            <label for="conv_2_out_channels">Conv 2 Out Channels:</label>
            <input type="number" id="conv_2_out_channels" name="conv_2_out_channels" value="8" required>
    
            <label for="conv_2_kernel">Conv 2 Kernel Size:</label>
            <input type="number" id="conv_2_kernel" name="conv_2_kernel" value="3" required>
    
            <label for="conv_2_padding">Conv 2 Padding:</label>
            <input type="number" id="conv_2_padding" name="conv_2_padding" value="1" required>
    
            <label for="MaxPool2d_2_kernel_size">MaxPool 2 Kernel Size:</label>
            <input type="number" id="MaxPool2d_2_kernel_size" name="MaxPool2d_2_kernel_size" value="2" required>
    
            <label for="fc_1_out_features">FC 1 Out Features:</label>
            <input type="number" id="fc_1_out_features" name="fc_1_out_features" value="512" required>
    
            <label for="hidden_size">Hidden Size:</label>
            <input type="number" id="hidden_size" name="hidden_size" value="32" required>
    
            <label for="num_layers">Num Layers:</label>
            <input type="number" id="num_layers" name="num_layers" value="1" required>
    
            <label for="dropout">Dropout:</label>
            <input type="number" step="0.01" id="dropout" name="dropout" value="0.2" required>

          </div>
        `;
        break;
      case "LeNet":
        container.innerHTML = `
            <div class="parameter-indent">
              <label for="conv_1_in_channels">Conv 1 In Channels:</label>
              <input type="number" id="conv_1_in_channels" name="conv_1_in_channels" value="19" required>
              <label for="conv_1_out_channels">Conv 1 Out Channels:</label>
              <input type="number" id="conv_1_out_channels" name="conv_1_out_channels" value="8" required>
              <!-- Add more parameters as needed -->
            </div>
          `;
        break;
      case "LSTM":
        container.innerHTML = `
            <div class="parameter-indent">
              <label for="hidden_size">Hidden Size:</label>
              <input type="number" id="hidden_size" name="hidden_size" value="32" required>
              <!-- Add more LSTM specific parameters -->
            </div>
          `;
        break;
      case "RNN":
        container.innerHTML = `
            <div class="parameter-indent">
              <label for="hidden_size">Hidden Size:</label>
              <input type="number" id="hidden_size" name="hidden_size" value="32" required>
              <!-- Add more RNN specific parameters -->
            </div>
          `;
        break;
      case "DNN":
        container.innerHTML = `
          <div class="parameter-indent">
            <label for="layers">Layers:</label>
            <input type="number" id="layers" name="layers" value="5" min="1" required>
          </div>
          <div id="layerParams"></div>
        `;
        setTimeout(() => {
          document
            .getElementById("layers")
            .addEventListener("change", generateLayerInputs);
          generateLayerInputs();
        }, 0);
        break;

      case "DummyClassifier":
        container.innerHTML = `
            <div class="parameter-indent">
              <p>This is a dummy classifier with no configurable parameters.</p>
            </div>
          `;
        break;
      default:
        console.error("Invalid model type!");
        break;
    }
  } else {
    console.error("Element #model_type or #model_config_container not found!");
  }
}

function generateLayerInputs() {
  const layers = document.getElementById("layers").value;
  const layerParamsDiv = document.getElementById("layerParams");

  // 清除旧的输入字段
  layerParamsDiv.innerHTML = "";

  // 为每一层生成一个输入字段
  for (let i = 1; i <= layers; i++) {
    const div = document.createElement("div");
    div.className = "parameter-indent";
    const label = document.createElement("label");
    label.htmlFor = `layer_${i}_units`;
    label.textContent = `Layer ${i} Units:`;
    const input = document.createElement("input");
    input.type = "number";
    input.id = `layer_${i}_units`;
    input.name = `layer_${i}_units`;
    input.value = "32"; // 默认值或者根据需要设置
    input.min = "1";
    input.required = true;

    div.appendChild(label);
    div.appendChild(input);
    layerParamsDiv.appendChild(div);
  }
}

document.addEventListener("DOMContentLoaded", function () {
  updateModelConfigFields();
});

var test_buy_signals;
var test_sell_signals;
var pred_buy_signals;
var pred_sell_signals;
var newest_buy_signals;
var newest_sell_signals;

$(document).ready(function () {
  $("#stockAnalysisForm").submit(function (event) {
    event.preventDefault();

    // Basic form fields
    var formObject = {
      start_date: $("#start_date").val(),
      stop_date: $("#stop_date").val(),
      stock_symbol: $("#stock_symbol").val(),
      test_indices: $("#test_indices").val(),
      training_symbols: $("#training_symbols").val(),
      split_ratio: parseFloat($("#split_ratio").val()),
      target_col: $("#target_col").val(),
      features_params: [],
      look_back: parseInt($("#look_back").val()),
      predict_steps: parseInt($("#predict_steps").val()),
      train_slide_steps: parseInt($("#train_slide_steps").val()),
      test_slide_steps: parseInt($("#test_slide_steps").val()),
      data_cleaning: {
        clean_type: $("#clean_type").val(),
        strategy: $("#strategy").val(),
      },
      model_type: $("#model_type").val(),
      model_params: {}, // This will need to be filled out based on selected model and its parameters
      training_epoch_num: parseInt($("#training_epoch_num").val()),
      online_training_epoch_num: parseInt(
        $("#online_training_epoch_num").val()
      ),
      learning_rate: parseFloat($("#learning_rate").val()),
      online_train_learning_rate: parseFloat(
        $("#online_train_learning_rate").val()
      ),
      patience: parseInt($("#patience").val()),
      min_delta: parseFloat($("#min_delta").val()),
      apply_weight: $("#apply_weight").val() === "True",
      data_update_mode: $("#data_update_mode").val(),
      filter: $("#filter").val(),
      trade_strategy: $("#trade_strategy").val(),
      filter_reverse_trend: $("#filter_reverse_trend").val(),
    };

    // Dynamically add features_params based on selected checkboxes
    $('.feature input[type="checkbox"]:checked').each(function () {
      var featureType = $(this).val();
      var featureId = $(this).attr("id");
      var parameters = {};

      $(`#params_${featureId} input, #params_${featureId} select`).each(
        function () {
          var paramName = $(this).attr("name").replace(`${featureId}_`, "");
          if (this.type === "number" || this.type === "text") {
            parameters[paramName] =
              this.type === "number"
                ? parseFloat($(this).val())
                : $(this).val();
          } else if (this.type === "date") {
            parameters[paramName] = $(this).val();
            parameters[paramName] = $.isNumeric(paramValue)
              ? parseFloat(paramValue)
              : paramValue;
          }
        }
      );

      formObject.features_params.push({
        type: featureType,
        ...parameters,
      });
    });

    $("input[id^='model_'], select[id^='model_']").each(function () {
      var param = $(this).attr("id").substring(6); // Remove "model_" prefix
      formObject.model_params[param] = $(this).val();
    });

    // Handle model specific parameters
    $("#model_config_container input, #model_config_container select").each(
      function () {
        var paramName = $(this).attr("name");
        if (this.type === "number" || this.type === "text") {
          formObject.model_params[paramName] =
            this.type === "number" ? parseFloat($(this).val()) : $(this).val();
        }
      }
    );

    var jsonData = JSON.stringify(formObject);

    $.ajax({
      type: "POST",
      url: "/saferTrader/revpred/run",
      contentType: "application/json",
      data: jsonData,
      success: function (response) {
        console.log(response.newest_buy_signals);
        console.log(response.newest_sell_signals);
        test_buy_signals = response.test_buy_signals;
        test_sell_signals = response.test_sell_signals;
        pred_buy_signals = response.pred_buy_signals;
        pred_sell_signals = response.pred_sell_signals;
        newest_buy_signals = response.newest_buy_signals;
        newest_sell_signals = response.newest_sell_signals;

        console.log("Form submitted successfully:", response);
        const container = $("#responseContainer");
        container.html("");
        // container.append('<div id="accordion"></div>');
        // const accordion = $("#accordion");

        // Message
        container.append(
          `<div class="alert alert-success"><strong>${response.msg}</strong></div>`
        );

        // Received Data
        const receivedData = response.receivedData;

        // Features Params
        let featuresParamsHtml = `<div class="card mt-3">
          <div class="card-header">Features Params</div>
          <div class="card-body">`;
        receivedData.features_params.forEach((feature, index) => {
          featuresParamsHtml += `<p class="card-text">Feature ${
            index + 1
          }: Type - ${feature.type}</p>`;
        });
        featuresParamsHtml += `</div></div>`;

        container.append(featuresParamsHtml);
        // Execution Time
        container.append(`<div class="card mt-3">
                <div class="card-header">Execution Time</div>
                <div class="card-body">
                  <p class="card-text">${response.execution_time} seconds</p>
                </div>
              </div>`);

        // Model Summary
        let summary = response.model_summary.replace(/\\n/g, "<br>");
        container.append(`
          <div class="card mt-3">
            <div class="card-header">
              Model Summary
            </div>
            <div id="collapseOne" class="collapse show" aria-labelledby="headingOne">
              <div class="card-body">
                ${summary}
              </div>
            </div>
          </div>
        `);

        // confusion_matrix_info
        const confusionMatrixData = JSON.parse(response.confusion_matrix_info); // Parse if it's a string

        // 创建表格的HTML字符串
        let tableHtml = `<div class="card mt-3">
        <div class="card-header">Confusion Matrix</div>
        <div class="card-body">
          <table class="table">
            <thead>
              <tr>
                <th></th>
                <th>Predicted Positive</th>
                <th>Predicted Negative</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th>Actual Positive</th>
                <td>${confusionMatrixData.TP}</td>
                <td>${confusionMatrixData.FP}</td>
              </tr>
              <tr>
                <th>Actual Negative</th>
                <td>${confusionMatrixData.FN}</td>
                <td>${confusionMatrixData.TN}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
      <div class="card mt-3">
        <div class="card-header">Performance Metrics</div>
        <div class="card-body">
          <table class="table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th>Accuracy</th>
                <td>${confusionMatrixData.Accuracy.toFixed(2)}</td>
              </tr>
              <tr>
                <th>Precision</th>
                <td>${confusionMatrixData.Precision.toFixed(2)}</td>
              </tr>
              <tr>
                <th>Recall</th>
                <td>${confusionMatrixData.Recall.toFixed(2)}</td>
              </tr>
              <tr>
                <th>F1 Score</th>
                <td>${confusionMatrixData["F1 Score"].toFixed(2)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>`;

        // confusion_matrix_info
        container.append(`<div class="card mt-3">
                <div class="card-header">Confusion Matrix</div>
                <div class="card-body">
                  ${tableHtml}
                  <div id="confusionMatrixHeatmap" style="width:100%; max-width:600px; margin:auto;"></div>
                </div>
              </div>`);

        // Prepare the data for the heatmap
        const data = [
          {
            z: [
              [confusionMatrixData.FN, confusionMatrixData.TN],
              [confusionMatrixData.TP, confusionMatrixData.FP],
            ],
            x: ["Predicted Positive", "Predicted Negative"],
            y: ["Actual Negative", "Actual Positive"],
            type: "heatmap",
            hoverongaps: false,
            colorscale: [
              [0, "#fee8c8"],
              [0.5, "#fdbb84"],
              [1, "#e34a33"],
            ],
            showscale: true,
          },
        ];

        // Create annotations for each cell of the heatmap
        const annotations = [];
        for (let i = 0; i < 2; i++) {
          for (let j = 0; j < 2; j++) {
            const color = "black";
            let result = "";
            if (i === 0 && j === 0) result = "TP";
            if (i === 0 && j === 1) result = "TN";
            if (i === 1 && j === 0) result = "FN";
            if (i === 1 && j === 1) result = "FP";
            annotations.push({
              x: j === 0 ? "Predicted Positive" : "Predicted Negative",
              y: i === 0 ? "Actual Negative" : "Actual Positive",
              text: result + ": " + data[0].z[i][j],
              font: {
                color: color,
              },
              showarrow: false,
            });
          }
        }

        // Define the layout for the heatmap
        const layout = {
          title: "Confusion Matrix Heatmap",
          annotations: annotations,
          xaxis: {
            title: "Predicted Value",
            side: "top",
          },
          yaxis: {
            title: "Actual Value",
          },
          margin: {
            l: 100,
            r: 0,
            b: 0,
            t: 150,
            pad: 4,
          },
          autosize: true,
        };

        // Plot the heatmap
        Plotly.newPlot("confusionMatrixHeatmap", data, layout);

        // Trading signal plots
        container.append(`<div class="card mt-3">
                <div class="card-header">Trading signal plots</div>
                <div class="card-body">    
                  <select class="signalTypeSelect">
                  <option value="pred">Predicted Signals</option>
                  <option value="test">Actual Signals</option>
                  </select>
                  <div id="Trading_signal_chart" class="Trading_signal_chart" style="height: 600px; min-width: 310px"></div>
                </div>
              </div>`);

        var selectedSignalType = $(".signalTypeSelect").last().val();
        loadChartData(selectedSignalType, "Trading_signal_chart");

        container.append(`
        <div class="card mt-3">
          <div class="card-header">Newest Trading signal</div>
          <div class="card-body">    
            <div id="Newest_Trading_signal" class="Trading_signal">
              <p>Buy Signal: ${newest_buy_signals
                .map(
                  (signal) =>
                    `Date: ${
                      new Date(signal[0]).toISOString().split("T")[0]
                    }, Signal: ${signal[1]}`
                )
                .join("<br>")}</p>
              <p>Sell Signal: ${newest_sell_signals
                .map(
                  (signal) =>
                    `Date: ${
                      new Date(signal[0]).toISOString().split("T")[0]
                    }, Signal: ${signal[1]}`
                )
                .join("<br>")}</p>
            </div>
          </div>
        </div>`);

        const TrendData = JSON.parse(response.newest_buy_trend);

        // Trading signal plots
        container.append(`<div class="card mt-3">
                <div class="card-header">Trading signal plots</div>
                <div class="card-body">    
                  <div id="Newest_trading_signal_chart" class="Newest_trading_signal_chart" style="height: 600px; min-width: 310px"></div>
                </div>
              </div>`);

        renderNewestTradeSignalChart(
          TrendData,
          newest_buy_signals,
          newest_sell_signals
        );

        const predDaysDifferenceResults = JSON.parse(
          response.pred_days_difference_results
        );
        const predDaysDifferenceAbsMean = JSON.parse(
          response.pred_days_difference_abs_mean
        );

        const dates = Object.keys(predDaysDifferenceResults.Date).map((key) =>
          Highcharts.dateFormat("%Y-%m-%d", predDaysDifferenceResults.Date[key])
        );
        const daysDifferences = Object.values(
          predDaysDifferenceResults.DaysDifference
        );

        container.append(`<div class="card mt-3">
            <div class="card-header">Predicted Days Difference Results</div>
            <div class="card-body">
              <div id="predDaysDifferenceAbsMean" class="predDaysDifferenceAbsMean">
                <p>Absolute Mean Days Difference: ${predDaysDifferenceAbsMean} Days</p>
              <div id="PredDaysDifferenceResultsChart" class="PredDaysDifferenceResultsChart" style="height: 600px; min-width: 310px"></div>
            </div>
          </div>`);

        Highcharts.chart("PredDaysDifferenceResultsChart", {
          chart: {
            type: "line",
          },
          title: {
            text: "Predicted Days Difference Results",
          },
          xAxis: {
            categories: dates,
            title: {
              text: "Date",
            },
          },
          yAxis: {
            title: {
              text: "Days Difference",
            },
          },
          series: [
            {
              name: "Days Difference",
              data: daysDifferences,
            },
          ],
          tooltip: {
            valueSuffix: " days",
          },
          plotOptions: {
            line: {
              dataLabels: {
                enabled: true,
              },
              enableMouseTracking: true,
            },
          },
        });

        const backtestingReport = JSON.parse(response.backtesting_report); // 假设这是一个对象，不是字符串
        console.log(backtestingReport);

        let reportHtml = `
        <div class="card mt-3">
            <div class="card-header">Backtesting Report</div>
            <div class="card-body">
                <h5>Sharpe Ratio</h5>
                <p>Sharpe Ratio: <strong>${
                  backtestingReport.sharpe_ratio.sharperatio
                }</strong></p>

                <h5>Drawdown</h5>
                <ul>
                    <li>Length: ${backtestingReport.drawdown.len}</li>
                    <li>Drawdown: ${backtestingReport.drawdown.drawdown}%</li>
                    <li>Money Down: $${
                      backtestingReport.drawdown.moneydown
                    }</li>
                    <li>Max Drawdown Length: ${
                      backtestingReport.drawdown.max.len
                    }</li>
                    <li>Max Drawdown: ${
                      backtestingReport.drawdown.max.drawdown
                    }%</li>
                    <li>Max Money Down: $${
                      backtestingReport.drawdown.max.moneydown
                    }</li>
                </ul>

                <h5>Trade Analyzer</h5>
                <ul>
                    <li>Total Trades: ${
                      backtestingReport.trade_analyzer.total.total
                    }</li>
                    <li>Open Trades: ${
                      backtestingReport.trade_analyzer.total.open
                    }</li>
                    <li>Closed Trades: ${
                      backtestingReport.trade_analyzer.total.closed
                    }</li>
                    <li>Won Trades: ${
                      backtestingReport.trade_analyzer.won.total
                    }</li>
                    <li>Lost Trades: ${
                      backtestingReport.trade_analyzer.lost.total
                    }</li>
                </ul>

                <h5>Profit and Loss</h5>
                <ul>
                    <li>Gross Total: $${backtestingReport.trade_analyzer.pnl.gross.total.toFixed(
                      2
                    )}</li>
                    <li>Net Total: $${backtestingReport.trade_analyzer.pnl.net.total.toFixed(
                      2
                    )}</li>
                    <li>Won Total: $${backtestingReport.trade_analyzer.won.pnl.total.toFixed(
                      2
                    )}</li>
                    <li>Lost Total: -$${Math.abs(
                      backtestingReport.trade_analyzer.lost.pnl.total
                    ).toFixed(2)}</li>
                </ul>

                <h5>Final Value</h5>
                <p>Final Portfolio Value: <strong>$${backtestingReport.final_value.toFixed(
                  2
                )}</strong></p>

                <h5>Performance Metrics</h5>
                <ul>
                    <li>PnL: $${backtestingReport.pnl.toFixed(2)}</li>
                    <li>PnL Percentage: ${backtestingReport.pnl_pct.toFixed(
                      2
                    )}%</li>
                    <li>Total Return: ${
                      backtestingReport.total_return.toFixed(2) * 100
                    }%</li>
                </ul>
            </div>
        </div>
        `;

        container.append(reportHtml);
      },
    });
  });
});

$(document).ready(function () {
  $(document).on("change", ".signalTypeSelect", function () {
    var selectedSignalType = $(this).val();
    var chartContainer = $(this).next(".Trading_signal_chart");
    // var chartId = chartContainer.attr("id");
    loadChartData(selectedSignalType);
  });
});

function loadChartData(signalType) {
  var formObject = {
    start_date: $("#start_date").val(),
    stop_date: $("#stop_date").val(),
    test_indices: $("#test_indices").val(),
    stock_symbol: $("#stock_symbol").val(),
  };
  console.log("Form data:", formObject);
  $.ajax({
    type: "post",
    url: "/saferTrader/revpred/get_history_data",
    data: JSON.stringify(formObject),
    contentType: "application/json",
    success: function (history_data_response) {
      history_data_response = JSON.parse(history_data_response);
      var chartData = {
        ticker: history_data_response.ticker,
        ohlc: history_data_response.ohlc,
        volume: history_data_response.volume,
        buy_signals:
          signalType === "test" ? test_buy_signals : pred_buy_signals,
        sell_signals:
          signalType === "test" ? test_sell_signals : pred_sell_signals,
        newest_buy_signals: newest_buy_signals,
        newest_sell_signals: newest_sell_signals,
      };

      renderHighchart(
        chartData.ticker,
        chartData.ohlc,
        chartData.volume,
        chartData.buy_signals,
        chartData.sell_signals,
        chartData.newest_buy_signals,
        chartData.newest_sell_signals
      );
    },
    error: function (xhr, status, error) {
      console.error("Error fetching data:", error);
    },
  });
}

function renderHighchart(
  ticker,
  ohlc_data,
  vol_data,
  buy_signals_data,
  sell_signals_data,
  newest_buy_data,
  newest_sell_data
) {
  const ohlc = [],
    volume = [],
    buy_signals = [],
    sell_signals = [],
    newest_buy_signals = [],
    newest_sell_signals = [],
    // split the data set into ohlc and volume
    dataLength = ohlc_data.length,
    buysignalsLength = buy_signals_data.length,
    sellsignalsLength = sell_signals_data.length,
    newestBuySignalsLength = newest_buy_data.length,
    newestSellSignalsLength = newest_sell_data.length;

  for (let i = 0; i < dataLength; i += 1) {
    ohlc.push([
      ohlc_data[i][0], // the date
      ohlc_data[i][1], // open
      ohlc_data[i][2], // high
      ohlc_data[i][3], // low
      ohlc_data[i][4], // close
    ]);
    volume.push([
      ohlc_data[i][0], // the date
      vol_data[i], // the volume
    ]);
  }
  for (let i = 0; i < buysignalsLength; i += 1) {
    buy_signals.push([buy_signals_data[i][0], buy_signals_data[i][1]]);
  }
  for (let i = 0; i < sellsignalsLength; i += 1) {
    sell_signals.push([sell_signals_data[i][0], sell_signals_data[i][1]]);
  }
  for (let i = 0; i < newestBuySignalsLength; i += 1) {
    newest_buy_signals.push([newest_buy_data[i][0], buy_signals_data[0][1]]);
  }
  for (let i = 0; i < newestSellSignalsLength; i += 1) {
    newest_sell_signals.push([newest_sell_data[i][0], sell_signals_data[0][1]]);
  }
  console.log(newest_buy_signals);
  console.log(newest_sell_signals);

  // create the chart
  Highcharts.stockChart("Trading_signal_chart", {
    rangeSelector: {
      selected: 4,
    },

    title: {
      text: `${ticker} Historical`,
    },

    xAxis: {
      type: "datetime",
      labels: {
        step: 1,
        formatter: function () {
          return Highcharts.dateFormat("%Y-%m-%d", this.value);
        },
      },
    },

    yAxis: [
      {
        labels: {
          align: "right",
          x: -3,
        },
        title: {
          text: "OHLC",
        },
        height: "60%",
        lineWidth: 2,
        resize: {
          enabled: true,
        },
      },
      {
        labels: {
          align: "right",
          x: -3,
        },
        title: {
          text: "Volume",
        },
        top: "65%",
        height: "35%",
        offset: 0,
        lineWidth: 2,
      },
    ],
    series: [
      {
        type: "candlestick",
        name: ticker,
        data: ohlc,
        tooltip: {
          pointFormatter: function () {
            const date = new Date(this.x);
            const dateString = Highcharts.dateFormat("%Y-%m-%d", date);
            const msg = ` symbol: <b>${ticker}</b>
                          <br>Date: <b>${dateString}</b>
                          <br>Open: <b>${this.open}</b>
                          <br>High: <b>${this.high}</b>
                          <br>Low: <b>${this.low}</b>
                          <br>Close: <b>${this.close}</b>`;
            return msg;
          },
        },
        color: "red",
        lineColor: "black",
        upColor: "green",
        upLineColor: "black",
      },
      {
        type: "column",
        name: "Volume",
        data: volume,
        yAxis: 1,
        tooltip: {
          pointFormatter: function () {
            const msg = `<br>Volume: <b>${this.y}</b>`;
            return msg;
          },
        },
      },
      {
        type: "scatter",
        data: buy_signals,
        name: "Buy",
        marker: {
          symbol: "triangle",
          fillColor: "green",
          lineColor: "green",
          lineWidth: 2,
          name: "Buy",
          enabled: true,
          radius: 6,
        },
        visibility: true,
        yAxis: 0,
        zIndex: 5,
        tooltip: {
          pointFormatter: function () {
            const date = new Date(this.x);
            const dateString = Highcharts.dateFormat("%Y-%m-%d", date);
            const msg = `Buy Signal: <b>${dateString}</b>
                            <br>Price: <b>${this.y}</b>`;
            return msg;
          },
        },
      },
      {
        type: "scatter",
        data: sell_signals,
        name: "Sell",
        marker: {
          symbol: "triangle-down",
          fillColor: "red",
          lineColor: "red",
          lineWidth: 2,
          name: "Sell",
          enabled: true,
          radius: 6,
        },
        visibility: true,
        yAxis: 0,
        zIndex: 5,
        tooltip: {
          pointFormatter: function () {
            const date = new Date(this.x);
            const dateString = Highcharts.dateFormat("%Y-%m-%d", date);
            const msg = `Sell Signal: <b>${dateString}</b>
                          <br>Price: <b>${this.y}</b>`;
            return msg;
          },
        },
      },
      {
        type: "scatter",
        data: newest_buy_signals,
        name: "Future Buy",
        marker: {
          symbol: "triangle",
          fillColor: "green",
          lineColor: "green",
          lineWidth: 2,
          name: "Buy",
          enabled: true,
          radius: 6,
        },
        visibility: true,
        yAxis: 0,
        zIndex: 5,
        tooltip: {
          pointFormatter: function () {
            const date = new Date(this.x);
            const dateString = Highcharts.dateFormat("%Y-%m-%d", date);
            const msg = `Buy Signal: <b>${dateString}</b>`;
            return msg;
          },
        },
      },
      {
        type: "scatter",
        data: newest_sell_signals,
        name: "Future Sell",
        marker: {
          symbol: "triangle-down",
          fillColor: "red",
          lineColor: "red",
          lineWidth: 2,
          name: "Sell",
          enabled: true,
          radius: 6,
        },
        visibility: true,
        yAxis: 0,
        zIndex: 5,
        tooltip: {
          pointFormatter: function () {
            const date = new Date(this.x);
            const dateString = Highcharts.dateFormat("%Y-%m-%d", date);
            const msg = `Sell Signal: <b>${dateString}</b>`;
            return msg;
          },
        },
      },
    ],
    tooltip: {
      split: true,
    },
  });
}

function renderNewestTradeSignalChart(TrendData, buySignals, sellSignals) {
  // Parse buy trend data
  const trendData = Object.entries(TrendData).map(([time, value]) => [
    parseInt(time),
    parseFloat(value), // 确保值是数字类型
  ]);

  // 创建一个便于查找的趋势数据映射（日期：值）
  const trendDataMap = new Map(trendData);

  // 准备买入信号数据，将y值设置为对应日期的趋势值
  const buySignalsData = buySignals.map(([time]) => ({
    x: parseInt(time),
    y: trendDataMap.get(parseInt(time)) || null, // 如果找不到对应日期，则设置为null
    marker: {
      symbol: 'triangle',
      fillColor: 'green',
      lineColor: 'green',
      lineWidth: 1,
      radius: 6,
    }
  }));

  // 准备卖出信号数据，同样将y值设置为对应日期的趋势值
  const sellSignalsData = sellSignals.map(([time]) => ({
    x: parseInt(time),
    y: trendDataMap.get(parseInt(time)) || null, // 如果找不到对应日期，则设置为null
    marker: {
      symbol: 'triangle-down',
      fillColor: 'red',
      lineColor: 'red',
      lineWidth: 1,
      radius: 6,
    }
  }));

  // 配置并创建Highcharts图表
  Highcharts.chart("Newest_trading_signal_chart", {
    chart: {
      type: "line",
    },
    title: {
      text: "Trend with Signals",
    },
    xAxis: {
      type: "datetime",
      dateTimeLabelFormats: {
        day: "%Y-%m-%d",
      },
      labels: {
        formatter: function () {
          return Highcharts.dateFormat("%Y-%m-%d", this.value);
        },
      },
    },
    yAxis: {
      title: {
        text: "Value",
      },
    },
    series: [
      {
        name: "Trend",
        data: trendData,
        id: "trend",
      },
      {
        type: "scatter",
        data: buySignalsData,
        name: "Buy Signals",
        marker: {
          symbol: "triangle",
          fillColor: "green",
          lineColor: "green",
          lineWidth: 2,
          radius: 6,
        },
        yAxis: 0,
        zIndex: 5,
      },
      {
        type: "scatter",
        data: sellSignalsData,
        name: "Sell Signals",
        marker: {
          symbol: "triangle-down",
          fillColor: "red",
          lineColor: "red",
          lineWidth: 2,
          radius: 6,
        },
        yAxis: 0,
        zIndex: 5,
      }
    ],
  });
}
