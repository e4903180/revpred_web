import json
from revpred.lib.reversePrediction.preprocessor.preprocessor_pytorch import Preprocessor
from revpred.lib.reversePrediction.model.model_pytorch import Model
from revpred.lib.reversePrediction.postprocessor.postprocessor import Postprocesser
from revpred.lib.reversePrediction.evaluator.evaluator_pytorch import Evaluator
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import os
import time
import pickle
from sklearn.model_selection import train_test_split

class ReversePrediction():
    def set_seed(self, seed_value):
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)
        random.seed(seed_value)
        torch.manual_seed(seed_value)


    def run(self, params): 
        self.set_seed(42)
        preprocessor = Preprocessor(params)
        X_train, y_train, X_val, y_val, X_test, y_test, test_dates, X_newest, x_newest_date, y_newest_date, target_symbol_data = preprocessor.get_multiple_data()
        
        start_time = time.time()
        model_wrapper = Model(params=params)
        model, history, y_preds, online_training_losses, online_training_acc = \
            model_wrapper.run(X_train, y_train, X_test, y_test, X_val, y_val)
        y_pred_newest = model.forward(X_newest)
        end_time = time.time()
        execution_time = end_time - start_time
        
        y_preds = torch.tensor(y_preds, dtype=torch.float32)
        y_preds = preprocessor.change_values_after_first_reverse_point(y_preds)
        y_pred_newest = torch.tensor(y_pred_newest, dtype=torch.float32)
        y_pred_newest = preprocessor.change_values_after_first_reverse_point(y_pred_newest)
        
        postprocessor = Postprocesser()
        X_train, X_test, X_newest, y_train, y_test, y_preds = \
            postprocessor.check_shape(X_train, X_test, X_newest, y_train, y_test, y_preds, reshape=params['model_params'][params['model_type']]['reshape'])
        test_trade_signals = postprocessor.process_signals(y_test, test_dates, False)
        pred_trade_signals = postprocessor.process_signals(y_preds, test_dates, params['filter'])
        newest_trade_signals = postprocessor.process_signals(y_pred_newest, y_newest_date, params['filter'])
        newest_trade_signals['up_trend'] = y_pred_newest.detach().numpy()[0, :, 0]
        newest_trade_signals['down_trend'] = y_pred_newest.detach().numpy()[0, :, 1]
        
        evaluator = Evaluator(params=params)
        model_summary, pred_days_difference_results, pred_days_difference_abs_mean, backtesting_report, trade_summary, execution_time, confusion_matrix_info, y_test, y_preds = evaluator.generate_numericale_data(model,
                        y_test, y_preds, target_symbol_data, pred_trade_signals, test_trade_signals, execution_time)
        
        return model_summary, pred_days_difference_results, pred_days_difference_abs_mean, backtesting_report, trade_summary, execution_time, confusion_matrix_info, y_test, y_preds, test_trade_signals, pred_trade_signals, newest_trade_signals

    def run_2(self, params):
        self.set_seed(42)
        preprocessor = Preprocessor(params)
        train_indices = preprocessor.params['train_indices']
        test_indice = preprocessor.params['test_indices']

        test_dataset = preprocessor.fetch_stock_data(test_indice, preprocessor.params['start_date'], preprocessor.params['stop_date'])
        for single_feature_params in preprocessor.params['features_params']:
            feature_type = single_feature_params["type"]
            test_dataset = preprocessor.add_feature(test_dataset, feature_type, **single_feature_params)
        test_dataset, issues_detected = preprocessor.add_data_cleaner(test_dataset,
            clean_type=preprocessor.params['data_cleaning']['clean_type'], strategy=preprocessor.params['data_cleaning']['strategy'])
        X_train, y_train, X_test, y_test, _, test_dates = \
            preprocessor.process_data(test_dataset, split_ratio=preprocessor.params['split_ratio'], target_col=preprocessor.params['target_col'],
                                    feature_cols=preprocessor.params['feature_cols'], look_back=preprocessor.params['look_back'],
                                    predict_steps=preprocessor.params['predict_steps'],
                                    train_slide_steps=preprocessor.params['train_slide_steps'],
                                    test_slide_steps=preprocessor.params['test_slide_steps'],
                                    reshape=preprocessor.params['model_params'][preprocessor.params['model_type']]['reshape'])
        X_newest, x_newest_date, y_newest_date = preprocessor.create_x_newest_data(test_dataset, preprocessor.params['look_back'])
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        target_symbol_data = test_dataset
        start_time = time.time()
        online_training_losses = []
        online_training_acc = []
        online_training_val_losses = []
        online_training_val_acc = []
        y_preds = []

        X_train_set = X_train
        y_train_set = y_train
        # params['training_epoch_num'] = 10
        for i in tqdm.tqdm(range(len(X_test))):
            model_wrapper = Model(params=params)
            input_shape = X_train.shape
            model = model_wrapper.create_model(model_wrapper.params['model_type'], input_shape)
            history, model = model_wrapper.train_model(model, X_train, y_train, X_val, y_val, model_wrapper.params['apply_weight'])
            y_pred = model_wrapper.infer_model(model, X_test[i:i+1])
            y_preds.append(y_pred[0])
            online_training_losses.append(history['loss'])
            online_training_acc.append(history['binary_accuracy'])
            online_training_val_losses.append(history['val_loss'])
            online_training_val_acc.append(history['val_binary_accuracy'])
            single_X_test = X_test[i:i+1]
            single_y_test = y_test[i:i+1]
            X_train_set = torch.cat((X_train_set[1:], single_X_test), dim=0)
            y_train_set = torch.cat((y_train_set[1:], single_y_test), dim=0)
            X_train, X_val, y_train, y_val = train_test_split(X_train_set, y_train_set, test_size=0.2, random_state=42)
            if preprocessor.params['filter_reverse_trend'] == "True":
                y_train = preprocessor.change_values_after_first_reverse_point(y_train)
                y_val = preprocessor.change_values_after_first_reverse_point(y_val)
                y_test = preprocessor.change_values_after_first_reverse_point(y_test)
        y_preds = torch.stack(y_preds).detach().numpy()
        end_time = time.time()
        execution_time = end_time - start_time
        y_preds = torch.tensor(y_preds, dtype=torch.float32)
        y_preds = preprocessor.change_values_after_first_reverse_point(y_preds)
        y_pred_newest = model.forward(X_newest)
        y_pred_newest = torch.tensor(y_pred_newest, dtype=torch.float32)
        y_pred_newest = preprocessor.change_values_after_first_reverse_point(y_pred_newest)
        postprocessor = Postprocesser()
        X_train, X_test, X_newest, y_train, y_test, y_preds = \
            postprocessor.check_shape(X_train, X_test, X_newest, y_train, y_test, y_preds, reshape=params['model_params'][params['model_type']]['reshape'])
        test_trade_signals = postprocessor.process_signals(y_test, test_dates, False)
        pred_trade_signals = postprocessor.process_signals(y_preds, test_dates, params['filter'])
        newest_trade_signals = postprocessor.process_signals(y_pred_newest, y_newest_date, False)
        evaluator = Evaluator(params=params)
        model_summary, pred_days_difference_results, pred_days_difference_abs_mean, backtesting_report, trade_summary, execution_time, confusion_matrix_info, y_test, y_preds = evaluator.generate_numericale_data(model,
                        y_test, y_preds, target_symbol_data, pred_trade_signals, test_trade_signals, execution_time)
        
        return model_summary, pred_days_difference_results, pred_days_difference_abs_mean, backtesting_report, trade_summary, execution_time, confusion_matrix_info, y_test, y_preds, test_trade_signals, pred_trade_signals, newest_trade_signals

