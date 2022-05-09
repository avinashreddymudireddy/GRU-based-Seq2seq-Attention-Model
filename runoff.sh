#!/bin/bash

python Final_api.py --model_train 1 --model_name_prefix run1 --master_df_path masterdf2.csv --location_name 434514  --up_stream_runoff_prediction_list 434478 399711 --Output_path ./ --past_history 96 --future_target 48 >> Runoffrun.log
