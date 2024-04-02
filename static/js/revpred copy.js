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

// document.addEventListener("DOMContentLoaded", function () {
//   const featuresList = [
//     { type: "Trend", order_days: 20, ma_days: 20, trend_days: 5 },
//     { type: "MACD", fast_period: 5, slow_period: 10, signal_period: 9 },
//     { type: "ROC", trend_days: 5 },
//     // Add other features from your list...
//   ];

//   const featuresSelectionContainer =
//     document.getElementById("features_selection");
//   const paramsContainer = document.getElementById("feature_params_container");

//   // Generate feature checkboxes
//   featuresList.forEach((feature, index) => {
//     const checkboxId = `feature_${feature.type.toLowerCase()}`;
//     const featureCheckboxHTML = `
//         <div class="feature">
//           <input
//             type="checkbox"
//             id="${checkboxId}"
//             name="feature_${feature.type}"
//             value="${feature.type}"
//             ${feature.type === "Trend" ? "checked" : ""}
//           />
//           <label for="${checkboxId}">${feature.type}</label>
//         </div>
//       `;
//     featuresSelectionContainer.innerHTML += featureCheckboxHTML;
//   });

//   // Handle feature selection change
//   document
//     .querySelectorAll('input[type="checkbox"][name^="feature_"]')
//     .forEach((checkbox) => {
//       checkbox.addEventListener("change", function () {
//         updateFeatureParams();
//       });
//     });

//   function updateFeatureParams() {
//     paramsContainer.innerHTML = ""; // Clear previous params
//     document
//       .querySelectorAll('input[type="checkbox"][name^="feature_"]:checked')
//       .forEach((checkedBox) => {
//         const featureType = checkedBox.value;
//         // Add specific parameters based on the selected feature
//         // Example for "Trend"
//         if (featureType === "Trend") {
//           paramsContainer.innerHTML += `
//             <div>
//               <h4>${featureType} Parameters:</h4>
//               <label>Order Days: <input type="number" name="${featureType}_order_days" value="20"></label>
//               <label>MA Days: <input type="number" name="${featureType}_ma_days" value="20"></label>
//               <label>Trend Days: <input type="number" name="${featureType}_trend_days" value="5"></label>
//             </div>
//           `;
//         }
//         // Handle other features similarly
//       });
//   }

//   // Initially update params for pre-checked features
//   updateFeatureParams();
// });

// document.addEventListener("DOMContentLoaded", function () {
//   // 示例：仅为 "Trend" 特征添加复选框和处理逻辑
//   const featureCheckboxHTML = `
//       <div class="feature">
//         <input
//           type="checkbox"
//           id="feature_trend"
//           name="feature_trend"
//           value="Trend"
//           checked
//         />
//         <label for="feature_trend">Trend</label>
//         <div class="feature_params" id="params_trend" style="margin-left: 20px;"></div>
//       </div>
//     `;

//   document.getElementById("features_selection").innerHTML +=
//     featureCheckboxHTML;

//   document
//     .getElementById("feature_trend")
//     .addEventListener("change", function () {
//       const trendParamsContainer = document.getElementById("params_trend");
//       if (this.checked) {
//         // 添加 "Trend" 特征的附属参数选择
//         trendParamsContainer.innerHTML = `
//           <label for="trend_option">Trend Method:</label>
//           <select name="trend_option" id="trend_option">
//             <option value="local_extrema">Local Extrema</option>
//             <option value="ma">MA</option>
//           </select>
//           <div id="trend_method_params"></div>
//         `;

//         document
//           .getElementById("trend_option")
//           .addEventListener("change", function () {
//             const methodParamsContainer = document.getElementById(
//               "trend_method_params"
//             );
//             if (this.value === "local_extrema") {
//               methodParamsContainer.innerHTML = `<label>Order Days: <input type="number" name="trend_local_extrema_order_days" value="20"></label>`;
//             } else if (this.value === "ma") {
//               methodParamsContainer.innerHTML = `
//               <label>MA Days: <input type="number" name="trend_ma_days" value="20"></label>
//               <label>Trend Days: <input type="number" name="trend_trend_days" value="5"></label>
//             `;
//             }
//           });

//         // 默认触发一次更改事件以显示初始附属参数
//         document
//           .getElementById("trend_option")
//           .dispatchEvent(new Event("change"));
//       } else {
//         trendParamsContainer.innerHTML = ""; // 清除参数选择
//       }
//     });
// });

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

// // document.addEventListener("DOMContentLoaded", (event) => {
// //   const container = document.getElementById("features_container"); // 确保这里定义了container

// //   // 确保在添加特征之前container已经定义
// //   if (!container) {
// //     console.error("Features container not found!");
// //     return;
// //   }

// //   featuresList.forEach((feature) => {
// //     const featureDiv = document.createElement("div");
// //     featureDiv.classList.add("feature");
// //     let featureHTML = `
// //             <div class="parameter-indent">
// //        //         `;

// //     // 根据特征类型添加特定的参数输入字段
// //     switch (feature.type) {
// //       case "Trend":
// //         featureHTML += `
// //                     <label>Order Days: <input type="number" name="order_days" value="${feature.order_days}"></label>
// //                     <label>MA Days: <input type="number" name="ma_days" value="${feature.ma_days}"></label>
// //                     <label>Trend Days: <input type="number" name="trend_days" value="${feature.trend_days}"></label>
// //                 `;
// //         break;
// //       case "MACD":
// //         featureHTML += `
// //                     <label>Fast Period: <input type="number" name="fast_period" value="${feature.fastperiod}"></label>
// //                     <label>Slow Period: <input type="number" name="slow_period" value="${feature.slowperiod}"></label>
// //                     <label>Signal Period: <input type="number" name="signal_period" value="${feature.signalperiod}"></label>
// //                 `;
// //         break;
// //       // 添加其他特征类型的处理逻辑
// //     }

// //     featureHTML += `
// //                 </div>
// //                 <button type="button" class="remove_feature_btn">Remove Feature</button>
// //             </div>
// //         `;
// //     featureDiv.innerHTML = featureHTML;
// //     container.appendChild(featureDiv);

// //     // 为移除按钮添加事件监听器
// //     featureDiv
// //       .querySelector(".remove_feature_btn")
// //       .addEventListener("click", function () {
// //         container.removeChild(featureDiv);
// //       });
// //   });

// //   document
// //     .querySelectorAll(".feature_type_select")
// //     .forEach((select) => select.dispatchEvent(new Event("change")));
// // });
// //  <label>Feature Type: ${feature.type}</label>
// //                 <div class="feature_params_container">

// document.addEventListener("DOMContentLoaded", (event) => {
//   document
//     .getElementById("add_feature_btn")
//     .addEventListener("click", function () {
//       const container = document.getElementById("features_container");
//       const featureDiv = document.createElement("div");
//       featureDiv.classList.add("feature");

//       let selectHTML = `
//             <select name="feature_type" class="feature_type_select">
//                 <option value="Trend">Trend</option>
//                 <option value="MACD">MACD</option>
//                 <option value="ROC">ROC</option>
//                 <option value="Stochastic Oscillator">Stochastic Oscillator</option>
//                 <option value="CCI">CCI</option>
//                 <option value="RSI">RSI</option>
//                 <option value="VMA">VMA</option>
//                 <option value="pctChange">pctChange</option>
//                 <option value="13W Treasury Yield">13W Treasury Yield</option>
//                 <option value="5Y Treasury Yield">5Y Treasury Yield</option>
//                 <option value="10Y Treasury Yield">10Y Treasury Yield</option>
//                 <option value="30Y Treasury Yield">30Y Treasury Yield</option>
//                 <option value="Bollinger Bands">Bollinger Bands</option>
//                 <option value="ATR">ATR</option>
//                 <option value="OBV">OBV</option>
//                 <option value="Parabolic SAR">Parabolic SAR</option>
//                 <option value="MOM">MOM</option>
//                 <option value="Williams %R">Williams %R</option>
//                 <option value="Chaikin MF">Chaikin MF</option>
//             </select>
//         `;

//       featureDiv.innerHTML = `
//             <div class="parameter-indent">
//                 ${selectHTML}
//                 <div class="feature_params_container"></div>
//                 <button type="button" class="remove_feature_btn">Remove Feature</button>
//                 <p></p>
//             </div>
//         `;
//       container.appendChild(featureDiv);

//       // Add change listener to the feature type select
//       const featureTypeSelect = featureDiv.querySelector(
//         ".feature_type_select"
//       );
//       featureTypeSelect.addEventListener("change", handleFeatureTypeChange);

//       // Add change listener to the feature type select
//       featureDiv
//         .querySelector(".feature_type_select")
//         .addEventListener("change", handleFeatureTypeChange);

//       // Add click listener to the remove feature button
//       featureDiv
//         .querySelector(".remove_feature_btn")
//         .addEventListener("click", function () {
//           container.removeChild(featureDiv);
//         });

//       // Trigger change event to load parameters immediately after adding the feature
//       featureTypeSelect.dispatchEvent(new Event("change"));

//       function handleFeatureTypeChange(event) {
//         const featureType = event.target.value;
//         const paramsContainer = event.target
//           .closest(".feature")
//           .querySelector(".feature_params_container");
//         paramsContainer.innerHTML = "";

//         switch (featureType) {
//           case "Trend":
//             paramsContainer.innerHTML += `
//                     <div class="parameter-indent">
//                         <label>Order Days: <input type="number" name="order_days" value="20"></label>
//                         <div>
//                             <label>Trend Method:</label>
//                             <select name="trend_method" class="trend_method_select">
//                                 <option value="Local Extrema">Local Extrema</option>
//                                 <option value="MA">MA</option>
//                             </select>
//                         </div>
//                         <div class="trend_method_params"></div>
//                     </div>
//                 `;

//             const trendMethodSelect = paramsContainer.querySelector(
//               ".trend_method_select"
//             );
//             trendMethodSelect.addEventListener("change", function () {
//               const method = this.value;
//               const methodParamsContainer = paramsContainer.querySelector(
//                 ".trend_method_params"
//               );
//               methodParamsContainer.innerHTML = "";

//               if (method === "Local Extrema") {
//                 methodParamsContainer.innerHTML = `
//                             <div class="parameter-indent">
//                                 <label>Order Days: <input type="number" name="order_days" value="20"></label>
//                             </div>
//                         `;
//               } else if (method === "MA") {
//                 methodParamsContainer.innerHTML = `
//                             <div class="parameter-indent">
//                                 <label>MA Days: <input type="number" name="ma_days" value="20"></label>
//                                 <label>Trend Days: <input type="number" name="trend_days" value="5"></label>
//                             </div>
//                         `;
//               }
//             });

//             trendMethodSelect.dispatchEvent(new Event("change"));
//             break;
//           case "MACD":
//             paramsContainer.innerHTML += `
//                         <div class="parameter-indent">
//                             <label>Fast Period: <input type="number" name="fast_period" value="5"></label>
//                             <label>Slow Period: <input type="number" name="slow_period" value="10"></label>
//                             <label>Signal Period: <input type="number" name="signal_period" value="9"></label>
//                         </div>
//                     `;
//             break;
//           case "ROC":
//             paramsContainer.innerHTML = `
//                         <div class="parameter-indent">
//                             <label>Time Period: <input type="number" name="time_period" value="14"></label>
//                         </div>
//                     `;
//           case "Stochastic Oscillator":
//             paramsContainer.innerHTML += `
//                         <div class="parameter-indent">
//                             <label>Trend Days: <input type="number" name="trend_days" value="5"></label>
//                         </div>
//                     `;
//             break;
//           case "CCI":
//             paramsContainer.innerHTML = `
//                         <div class="parameter-indent">
//                             <label>Time Period: <input type="number" name="time_period" value="14"></label>
//                         </div>
//                     `;
//             break;
//           case "RSI":
//             paramsContainer.innerHTML = `
//                         <div class="parameter-indent">
//                             <label>Time Period: <input type="number" name="time_period" value="14"></label>
//                         </div>
//                     `;
//             break;
//           case "VMA":
//             paramsContainer.innerHTML = `
//                         <div class="parameter-indent">
//                             <label>Time Period: <input type="number" name="time_period" value="14"></label>
//                         </div>
//                     `;
//             break;
//           case "pctChange":
//             break;
//           case "13W Treasury Yield":
//             paramsContainer.innerHTML += `
//                         <div class="parameter-indent">
//                             <label>Start Date: <input type="date" name="start_date" value="2001-01-01"></label>
//                             <label>End Date: <input type="date" name="end_date" value="2021-01-01"></label>
//                         </div>
//                     `;
//             break;
//           case "5Y Treasury Yield":
//             paramsContainer.innerHTML += `
//                         <div class="parameter-indent">
//                             <label>Start Date: <input type="date" name="start_date" value="2001-01-01"></label>
//                             <label>End Date: <input type="date" name="end_date" value="2021-01-01"></label>
//                         </div>
//                     `;
//             break;
//           case "10Y Treasury Yield":
//             paramsContainer.innerHTML += `
//                         <div class="parameter-indent">
//                             <label>Start Date: <input type="date" name="start_date" value="2001-01-01"></label>
//                             <label>End Date: <input type="date" name="end_date" value="2021-01-01"></label>
//                         </div>
//                     `;
//             break;
//           case "30Y Treasury Yield":
//             paramsContainer.innerHTML += `
//                         <div class="parameter-indent">
//                             <label>Start Date: <input type="date" name="start_date" value="2001-01-01"></label>
//                             <label>End Date: <input type="date" name="end_date" value="2021-01-01"></label>
//                         </div>
//                     `;
//             break;
//           case "Bollinger Bands":
//             paramsContainer.innerHTML += `
//                         <div class="parameter-indent">
//                             <label>Time Period: <input type="number" name="time_period" value="20"></label>
//                             <label>NB Dev Up: <input type="number" name="nbdevup" value="2"></label>
//                             <label>NB Dev Down: <input type="number" name="nbdevdn" value="2"></label>
//                         </div>
//                     `;
//             break;
//           case "ATR":
//             paramsContainer.innerHTML = `
//                         <div class="parameter-indent">
//                             <label>Time Period: <input type="number" name="time_period" value="14"></label>
//                         </div>
//                     `;
//             break;
//           case "Parabolic SAR":
//             paramsContainer.innerHTML += `
//                         <div class="parameter-indent">
//                             <label>Time Period: <input type="number" name="time_period" value="20"></label>
//                             <label>NB Dev Up: <input type="number" name="nbdevup" value="2"></label>
//                             <label>NB Dev Down: <input type="number" name="nbdevdn" value="2"></label>
//                         </div>
//                     `;
//             break;
//           case "MOM":
//             paramsContainer.innerHTML = `
//                         <div class="parameter-indent">
//                             <label>Time Period: <input type="number" name="time_period" value="10"></label>
//                         </div>
//                     `;
//             break;
//           case "Williams %R":
//             paramsContainer.innerHTML = `
//                         <div class="parameter-indent">
//                             <label>Time Period: <input type="number" name="lookback_period" value="14"></label>
//                         </div>
//                     `;
//             break;
//           case "Chaikin MF":
//             paramsContainer.innerHTML = `
//                         <div class="parameter-indent">
//                             <label>Time Period: <input type="number" name="time_period" value="20"></label>
//                         </div>
//                     `;
//             break;
//           // Add more cases for other feature types...
//           default:
//             console.log(
//               "Selected feature type does not have additional parameters."
//             );
//             break;
//         }
//       }
//     });

//   // Initialize feature type selects
//   document
//     .querySelectorAll(".feature_type_select")
//     .forEach((select) => select.dispatchEvent(new Event("change")));
// });

function updateModelConfigFields() {
  const modelTypeSelect = document.getElementById("model_type");
  const container = document.getElementById("model_config_container");

  if (modelTypeSelect && container) {
    const modelType = modelTypeSelect.value;
    container.innerHTML = "";

    if (modelType === "CNN_LSTM") {
      container.innerHTML = `
                <div class="parameter-indent">
                    <label for="look_back">Look Back:</label>
                    <input type="number" id="look_back" name="look_back" value="32" required>
                    
                </div>
            `;
    }
  } else {
    console.error("Element #model_type or #model_config_container not found!");
  }
}

document.addEventListener("DOMContentLoaded", function () {
  updateModelConfigFields();
});

var test_buy_signals;
var test_sell_signals;
var pred_buy_signals;
var pred_sell_signals;

$(document).ready(function () {
  $("#stockAnalysisForm").submit(function (event) {
    event.preventDefault();

    // Basic form fields
    var formObject = {
      start_date: $("#start_date").val(),
      stop_date: $("#stop_date").val(),
      stock_symbol: $("#stock_symbol").val(),
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
        test_buy_signals = response.test_buy_signals;
        test_sell_signals = response.test_sell_signals;
        pred_buy_signals = response.pred_buy_signals;
        pred_sell_signals = response.pred_sell_signals;
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
        // let receivedDataHtml = `<div class="card mt-3">
        //   <div class="card-header">Received Data</div>
        //   <div class="card-body">
        //     <p class="card-text">Start Date: ${receivedData.start_date}</p>
        //     <p class="card-text">Stop Date: ${receivedData.stop_date}</p>
        //     <p class="card-text">Stock Symbol: ${receivedData.stock_symbol}</p>
        //   </div>
        // </div>`;
        // container.append(receivedDataHtml);

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
            <div class="card-header">Confusion Matrix and Performance Metrics</div>
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
                  <tr>
                    <th>Accuracy</th>
                    <td colspan="2">${confusionMatrixData.Accuracy.toFixed(2)}</td>
                  </tr>
                  <tr>
                    <th>Precision</th>
                    <td colspan="2">${confusionMatrixData.Precision.toFixed(2)}</td>
                  </tr>
                  <tr>
                    <th>Recall</th>
                    <td colspan="2">${confusionMatrixData.Recall.toFixed(2)}</td>
                  </tr>
                  <tr>
                    <th>F1 Score</th>
                    <td colspan="2">${confusionMatrixData["F1 Score"].toFixed(
                      2
                    )}</td>
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
              [confusionMatrixData.TP, confusionMatrixData.FP]
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

        const predDaysDifferenceResults = JSON.parse(
          response.pred_days_difference_results
        );

        // 准备图表所需的数据
        const dates = Object.keys(predDaysDifferenceResults.Date).map((key) =>
          Highcharts.dateFormat("%Y-%m-%d", predDaysDifferenceResults.Date[key])
        );
        const daysDifferences = Object.values(
          predDaysDifferenceResults.DaysDifference
        );

        container.append(`<div class="card mt-3">
            <div class="card-header">Predicted Days Difference Results</div>
            <div class="card-body">
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
  console.log("Document ready!");
});

function loadChartData(signalType) {
  var formObject = {
    start_date: $("#start_date").val(),
    stop_date: $("#stop_date").val(),
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
      };

      renderHighchart(
        chartData.ticker,
        chartData.ohlc,
        chartData.volume,
        chartData.buy_signals,
        chartData.sell_signals
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
  sell_signals_data
) {
  const ohlc = [],
    volume = [],
    buy_signals = [],
    sell_signals = [],
    // split the data set into ohlc and volume
    dataLength = ohlc_data.length,
    buysignalsLength = buy_signals_data.length,
    sellsignalsLength = sell_signals_data.length;
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
        color: "red", // 下跌的蜡烛填充色（红色）
        lineColor: "black", // 下跌的蜡烛轮廓线颜色（红色）
        upColor: "green", // 上涨的蜡烛填充色（绿色）
        upLineColor: "black", // 上涨的蜡烛轮廓线颜色（绿色），如果不设置，将使用lineColor的值
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
    ],
    tooltip: {
      split: true,
    },
  });
}
