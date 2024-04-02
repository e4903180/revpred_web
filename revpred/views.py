from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from revpred.lib.reversePrediction.reversePrediction import ReversePrediction

import json
import yfinance as yf
import pandas as pd
import numpy as np
# Create your views here.


def web(request):
    return render(request, 'revpred/revpred.html', {})


def get_signals(row_trade_signals, history_data=None):
    row_trade_signals['Buy'] = pd.Series(index=row_trade_signals.index)
    row_trade_signals['Sell'] = pd.Series(index=row_trade_signals.index)
    row_trade_signals['Signal'] = row_trade_signals['Signal'].replace('Sell (first)', 'Sell')
    row_trade_signals['Signal'] = row_trade_signals['Signal'].replace('Sell (last)', 'Sell')
    row_trade_signals['Signal'] = row_trade_signals['Signal'].replace('Sell (first) (last)', 'Sell')
    row_trade_signals['Signal'] = row_trade_signals['Signal'].replace('Buy (first)', 'Buy')
    row_trade_signals['Signal'] = row_trade_signals['Signal'].replace('Buy (last)', 'Buy')
    row_trade_signals['Signal'] = row_trade_signals['Signal'].replace('Buy (first) (last)', 'Buy')
    row_trade_signals.index = pd.to_datetime(row_trade_signals['Date'])
    if history_data is not None:
        history_data['Date'] = history_data.index
        for i in row_trade_signals.index:
            if row_trade_signals['Signal'][i] == 'Buy':
                row_trade_signals.loc[i, 'Buy'] = history_data.loc[i, 'Low']
            elif row_trade_signals['Signal'][i] == 'Sell':
                row_trade_signals.loc[i, 'Sell'] = history_data.loc[i, 'High']
    else:
        for i in row_trade_signals.index:
            if row_trade_signals['Signal'][i] == 'Buy':
                row_trade_signals.loc[i, 'Buy'] = 'Buy'
            elif row_trade_signals['Signal'][i] == 'Sell':
                row_trade_signals.loc[i, 'Sell'] = 'Sell'
    buy_signals = row_trade_signals.dropna(subset=['Buy'])
    sell_signals = row_trade_signals.dropna(subset=['Sell'])
    buy_signals['Date'] = pd.to_datetime(buy_signals.index)
    buy_signals['Date'] = buy_signals['Date'].apply(lambda x: int(x.timestamp() * 1000))
    buy_signals = buy_signals[['Date', 'Buy']].values.tolist()
    sell_signals['Date'] = pd.to_datetime(sell_signals.index)
    sell_signals['Date'] = sell_signals['Date'].apply(lambda x: int(x.timestamp() * 1000))
    sell_signals = sell_signals[['Date', 'Sell']].values.tolist()
    return buy_signals, sell_signals

@csrf_exempt
def run(request):
    try:
        # data = json.loads(request.body)
        # with open('file.json', 'w') as f:
        #     json.dump(data, f)
        # ticker = data['stock_symbol']
        # start_date = data['start_date']
        # stop_date = data['stop_date']
        # with open('parameters.json', 'r') as f:
        #     params = json.load(f)

        # rp = ReversePrediction()
        # model_summary, pred_days_difference_results, pred_days_difference_abs_mean, backtesting_report, trade_summary, execution_time, confusion_matrix_info, y_test, y_preds, test_trade_signals, pred_trade_signals, newest_trade_signals = rp.run(
        #     params)
        # confusion_matrix_info_json = json.dumps(confusion_matrix_info)
        # model_summary_json = json.dumps(model_summary)
        # backtesting_report_json = json.dumps(backtesting_report)
        # pred_days_difference_results_json = pred_days_difference_results.to_json()
        # pred_days_difference_abs_mean_json = json.dumps(pred_days_difference_abs_mean)
        # trade_summary_json = json.dumps(trade_summary)
        # execution_time_json = json.dumps(execution_time)
        
        # history_data = yf.download(ticker, start=start_date, end=stop_date) 
        # history_data.to_csv(f'history_data_{ticker}_{start_date}_{stop_date}_0.csv')
        # test_buy_signals, test_sell_signals = get_signals(test_trade_signals, history_data)
        # pred_buy_signals, pred_sell_signals = get_signals(pred_trade_signals, history_data)
        # newest_buy_signals, newest_sell_signals = get_signals(newest_trade_signals)
        # newest_trade_signals.to_csv('newest_trade_signals.csv')
        # newest_buy_trend = newest_trade_signals['up_trend'].to_json()
        
        # # test_trade_signals.to_csv('test_trade_signals.csv')
        # # pd.DataFrame(test_buy_signals).to_csv('test_buy_signals.csv')
        # # pd.DataFrame(test_sell_signals).to_csv('test_sell_signals.csv')
        # # pred_trade_signals.to_csv('pred_trade_signals.csv')
        # # pd.DataFrame(pred_buy_signals).to_csv('pred_buy_signals.csv')
        # # pd.DataFrame(pred_sell_signals).to_csv('pred_sell_signals.csv')
        
        # response = {
        #     'msg': 'Received!',
        #     'receivedData': data,
        #     'usingData': params,
        #     'confusion_matrix_info': confusion_matrix_info_json,
        #     'model_summary': model_summary_json,
        #     'backtesting_report': backtesting_report_json,
        #     'pred_days_difference_results': pred_days_difference_results_json,
        #     'pred_days_difference_abs_mean': pred_days_difference_abs_mean_json,
        #     'trade_summary': trade_summary_json,
        #     'execution_time': execution_time_json,
        #     'test_buy_signals': test_buy_signals,
        #     'test_sell_signals': test_sell_signals,
        #     'pred_buy_signals': pred_buy_signals,
        #     'pred_sell_signals': pred_sell_signals,
        #     'newest_buy_signals': newest_buy_signals,
        #     'newest_sell_signals': newest_sell_signals,
        #     'newest_buy_trend': newest_buy_trend,
        # }
        # with open('summary.json', 'w') as f:
        #     json.dump(response, f)
            
        # with open('summary_training.json', 'r') as f:
        #     response = json.load(f)
        with open('summary.json', 'r') as f:
            response = json.load(f)
        return JsonResponse(response, status=200, safe=False)
    except json.JSONDecodeError as e:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)


@csrf_exempt
def get_history_data(request):
    try:
        data = json.loads(request.body)
        ticker = data.get('stock_symbol')
        start_date = data.get('start_date')
        stop_date = data.get('stop_date')
        history_data = yf.download(ticker, start=start_date, end=stop_date)
        history_data.to_csv(f'history_data_{ticker}_{start_date}_{stop_date}_1.csv')
        history_data['Date'] = history_data.index
        history_data['Date'] = history_data['Date'].apply(lambda x: int(x.timestamp() * 1000))
        # Convert history_data to Highcharts format
        ohlc = history_data[['Date', 'Open', 'High', 'Low', 'Close']].values.tolist()
        volume = history_data['Volume'].values.tolist()
        response = {
            'ticker': ticker,
            'ohlc': ohlc,
            'volume': volume,
        }
        response = json.dumps(response)
        return JsonResponse(response, status=200, safe=False)
    except json.JSONDecodeError as e:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    