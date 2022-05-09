#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import gc
import sys
import pickle
import argparse
from sklearn.preprocessing import StandardScaler
import scipy.stats as st
from scipy.stats import sem, t
from scipy import mean
import joblib
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

print(f"Tensorflow Version: {tf.__version__}")
print(f"Pandas Version: {pd.__version__}")
print(f"Numpy Version: {np.__version__}")
print(f"System Version: {sys.version}")
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from keras.regularizers import l1_l2, l1, l2
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

def computCF(data, confidence):
    n = len(data)
    m = mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    return(m - h, m+h)
def Interpolate(pdv, r1, r2):

    first_v = pdv[0, :]
    last_v = pdv[-1, :]


    ft_v = first_v
    for i in range(r1-1):
        ft_v = np.vstack((ft_v, first_v))
    lt_v = last_v
    for i in range((2*r2)-1):
        lt_v = np.vstack((lt_v, last_v))
    #print(ft_a.shape, pdt.shape, lt_a.shape, ft_v.shape, pdv.shape, lt_a.shape)
    final = np.vstack((ft_v, np.vstack((pdv, lt_v))))
    return final

def get_df(ds, sc, preds, location):
    dims = preds.shape[1]
    final =np.array([])
    for i in range(dims):
        pds = preds[:, i].tolist()
        ds_c = ds.copy()
        ds_c[:,-1]= pds

        ds_c2 = sc.inverse_transform(ds_c)
        #print(max(ds_c2[:, 1]))
        if i == 0:
            final = np.array(ds_c2[:,-1])
        else:
            final = np.column_stack((final, np.array(ds_c2[:,-1])))
    ds_c = ds.copy()
    ds_c2 = sc.inverse_transform(ds_c)

    final = np.column_stack((np.array(ds_c2[:,0]), np.column_stack((np.array(ds_c2[:,1]), np.column_stack((np.array(ds_c2[:,2]), final))))))
    colnames = ["up_"+location, "p_"+location, 'original_flow_'+location]
    for i in range(dims):
        lag = "lag_"+str(i+1)
        colnames.append(lag)
    df = pd.DataFrame(data=final, columns=colnames)

    return df
def get_df2(ds, sc, preds, location):
    dims = preds.shape[1]
    final =np.array([])
    for i in range(dims):
        pds = preds[:, i].tolist()
        ds_c = ds.copy()
        ds_c[:,-1]= pds

        ds_c2 = sc.inverse_transform(ds_c)
        #print(max(ds_c2[:, 1]))
        if i == 0:
            final = np.array(ds_c2[:,-1])
        else:
            final = np.column_stack((final, np.array(ds_c2[:,-1])))
    ds_c = ds.copy()
    ds_c2 = sc.inverse_transform(ds_c)

    final = np.column_stack((np.array(ds_c2[:,0]), np.column_stack((np.array(ds_c2[:,1]), np.column_stack((np.array(ds_c2[:,2]), np.column_stack((np.array(ds_c2[:,3]), final))))))))
    colnames = ["up2_"+location,"up1_"+location, "p_"+location, 'original_flow_'+location]
    for i in range(dims):
        lag = "lag_"+str(i+1)
        colnames.append(lag)
    df = pd.DataFrame(data=final, columns=colnames)

    return df

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data1 = []
    data2 = []
    data3 = []
    labels = []

    end_index_copy = end_index
    start_index = start_index + history_size

    if end_index is not None:
        print("start_index",start_index)
        print("end_index",end_index)
        for i in range(start_index, end_index-target_size):
            indices1 = range(i-history_size, i+target_size, step)
            indices2 = range(i-history_size, i, step)
            indices3 = range(i-target_size, i, step)
            data1.append(dataset[indices1, :-1])
            data2.append(dataset[indices2, -1:])
            data3.append(dataset[indices3, -1:])
            if single_step:
                labels.append(target[i+target_size])
            else:
                labels.append(target[i:i+target_size])
        print("training dataset done")
    else:
        end_index = len(dataset) - target_size
        print("start_index",start_index)
        print("end_index",end_index)
        for i in range(start_index, end_index-target_size):
            indices1 = range(i-history_size, i+target_size, step)
            indices2 = range(i-history_size, i, step)
            indices3 = range(i-target_size, i, step)
            data1.append(dataset[indices1, :-1])
            data2.append(dataset[indices2, -1:])
            data3.append(dataset[indices3, -1:])

            if single_step:
                labels.append(target[i+target_size])
            else:
                labels.append(target[i:i+target_size])

    return np.array(data1), np.array(data2), np.array(data3), np.array(labels)

class WeightedAverage(tf.keras.layers.Layer):
    def __init__(self, n_output):
        super(WeightedAverage, self).__init__()
        self.n_output = n_output
    def build(self, input_shape):
        self.W = self.add_weight("kernel",
                                  shape=[1,1,self.n_output])

    def call(self, inputs):

        # inputs is a list of tensor of shape [(n_batch, n_feat), ..., (n_batch, n_feat)]
        # expand last dim of each input passed [(n_batch, n_feat, 1), ..., (n_batch, n_feat, 1)]
        inputs = [tf.expand_dims(i, -1) for i in inputs]
        inputs = Concatenate(axis=-1)(inputs) # (n_batch, n_feat, n_inputs)
        weights = tf.nn.softmax(self.W, axis=-1) # (1,1,n_inputs)
        # weights sum up to one on last dim

        return tf.reduce_sum(weights*inputs, axis=-1)
    def get_config(self):
        return {"n_output": self.n_output}




def attention_GRU_model(x1_inp_shape, x2_inp_shape,units, opt, ft):
    train_data_multi_u_p = tf.keras.Input(shape=x1_inp_shape, name='train_data_multi_u_p')
    encoder1 = tf.keras.layers.GRU(units, return_state=True, return_sequences=True, kernel_regularizer =tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01))
    encoder1_outputs, state_c1 = encoder1(train_data_multi_u_p)

    train_data_multi_s = tf.keras.Input(shape=x2_inp_shape, name='train_data_multi_s')
    encoder2 = tf.keras.layers.GRU(units, return_state=True, return_sequences=True, kernel_regularizer =tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01))
    encoder2_outputs, state_c2 = encoder2(train_data_multi_s)

    x_outs = tf.keras.layers.Concatenate(axis=1)([encoder1_outputs, encoder2_outputs])
    #x_c = tf.keras.layers.Add()([state_c1, state_c2])
    x_c = WeightedAverage(n_output=2)([state_c1, state_c2])
    #x_c = tf.keras.layers.Lambda(WeightedAverage(n_output=2), name="lambda_layer1")([state_c1, state_c2])
    print('x_c',x_c)
    #x_c = WeightedAverage()([state_c1, state_c2])
    decoder_input = RepeatVector(ft)(x_c)

#     print("decoder_input \n", decoder_input, "\n")

    decoder_out, decoder_state = tf.keras.layers.GRU(units, return_state=True, return_sequences=True, kernel_regularizer =tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01))(decoder_input, initial_state= x_c)
    print("decoder_out \n", decoder_out, "\n")
    attention = tf.keras.layers.dot([decoder_out, x_outs], axes=[2, 2])
    attention = tf.keras.layers.Activation('softmax')(attention)

    context = tf.keras.layers.dot([attention, x_outs], axes=[2,1])
    x = WeightedAverage(n_output=2)([context, decoder_out])
    #x = tf.keras.layers.Lambda(WeightedAverage(n_output=2), name="lambda_layer2")([context, decoder_out])
    #x = WeightedAverage()([context, decoder_out])
    #x = tf.keras.layers.Add()([context, decoder_out])

    output = tf.keras.layers.Dense(1, activation='linear')(x)
    model = tf.keras.models.Model(inputs=[train_data_multi_u_p, train_data_multi_s], outputs=output)

    config = model.get_config()

    new_model = tf.keras.models.Model.from_config(config, custom_objects={"WeightedAverage": WeightedAverage})

    new_model.compile(loss='mse', optimizer=opt, metrics=["mse"])
    return new_model

def stitch_preds(pt, pa, r1, r2, r3):
    pts = pt[:,0].tolist()
    pas = pa[:,0].tolist()
    partssum = (pts[0]-0)/r1
    first = []
    s = 0
    for i in range(r1):
        first.append(s)
        s = s+partssum
    partssum2 = (pas[0]-pts[-1])/r2
    second = []
    s = pts[-1]
    for i in range(r2):
        second.append(s)
        s = s+partssum2
    partssum3 = (pas[-1]-pas[-3])/2
    third = []
    s = pas[-1]
    for i in range(r3):
        third.append(s)
        s = s+partssum3

    final_preds = first + pts + second + pas + third
    return final_preds

def getpreds_general(Pastval, model, ds, TYPE):
    preds= np.array([])
    flag = 0
    if TYPE == 'all':
        pred = model.predict([ds[0][-Pastval:], ds[1][-Pastval:]])
        print(pred.shape)
        #print("pred", pred)
        preds = pred[:, :, 0]
        #print("preds", preds)
    else:
        pred = model.predict(np.array(ds[-Pastval:]))
        #print("pred", pred)
        preds = pred[:, :, 0]
        #print("preds", preds)
    print("done")
    return preds



def model_training(master_df, location_name, up_stream_runoff_prediction_list,
        Output_path, past_history, future_target, STEP = 1, EPOCHS = 50,
        BATCH_SIZE = 512, LR = 0.0001, CELLS= 512, model_name_prefix = ""):

    Scalar = StandardScaler()

    #scalar_filename = "scalar_m_"+model_name_prefix+".save"
    #Scalar = joblib.load(scalar_filename)


    to_save_path = Output_path+ str(location_name)+ "_" + model_name_prefix
    if not os.path.exists(to_save_path):

        os.makedirs(to_save_path)

    data = master_df[['p_'+str(location_name),'s_'+str(location_name)]]
    print(to_save_path)
    to_save_path = Output_path+ str(location_name)+ "_" + model_name_prefix + "/"+ model_name_prefix + "_"
    print(to_save_path)
    scalar_filename = to_save_path+"scalar.pkl"
    with open(scalar_filename, 'wb') as f:
        pickle.dump(Scalar, f)

    if len(up_stream_runoff_prediction_list) == 0:
        data['up_'+str(location_name)] = 0.0
    else:
        for inx, v in enumerate(up_stream_runoff_prediction_list):
            data['up_'+str(location_name)+"_"+str(inx+1)] = v
            cols1 = data.columns.tolist()
            cols1 = cols1[-1:] + cols1[:-1]
            data = data[cols1]
    scalar = Scalar.fit(data.values)
    dataset= scalar.transform(data.values)

    TRAIN_SPLIT = int((4*len(master_df))/5)


    future_target = future_target * STEP
    #future_target = int(futures[0]) * STEP


    x1_train, x2_train, d_train, y_train = multivariate_data(dataset, dataset[:, -1], 0,
                                                     TRAIN_SPLIT, past_history,
                                                     future_target, STEP)
    x1_val, x2_val, d_val, y_val = multivariate_data(dataset, dataset[:, -1],
                                                 TRAIN_SPLIT, None, past_history,
                                                 future_target, STEP)



    #opt =  tf.keras.optimizers.RMSprop(lr=0.001)

    opt = tf.keras.optimizers.Adam(
        learning_rate= LR, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam', decay= 1e-4, clipvalue=1.0)

    model_attention_GRU = attention_GRU_model(x1_train.shape[-2:], x2_train.shape[-2:], CELLS, opt, future_target)
    valsplit = 0.2

    TRAIN_EVALUATION_INTERVAL = len(x1_train)*(1-valsplit)//BATCH_SIZE
    #VAL_EVALUATION_INTERVAL = len(x1_val_multi_mini)//BATCH_SIZE



    early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.0001, verbose=1, patience=40)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=15, cooldown=30, min_lr=1e-8)

    multi_step_history_attention_GRU = model_attention_GRU.fit([x1_train, x2_train], y_train,
                                              epochs=EPOCHS,
                                              shuffle=True,
                                              batch_size=BATCH_SIZE,
                                              validation_split=valsplit,
                                              verbose = 2,
                                              callbacks=[early_stopping, reduce_lr])
    #tf.keras.models.save_model(model_attention_GRU, to_save_path+'model_attention_GRU.h5')
    tf.keras.models.save(model_attention_GRU, to_save_path+'model_attention_GRU')
    preds_at_val =  getpreds_general(len(x1_val), model_attention_GRU, [x1_val, x2_val],'all');
    preds_at_train =  getpreds_general(len(x1_train), model_attention_GRU, [x1_train, x2_train],'all');

    #Saving important data
    with open(to_save_path+'preds_at_val.pkl', 'wb') as f:
        pickle.dump(preds_at_val, f)


    with open(to_save_path+'preds_at_train.pkl', 'wb') as f:
        pickle.dump(preds_at_train, f)

    pickle.dump(x1_train, open(to_save_path+"x1_train.pkl", 'wb'), protocol=4)
    pickle.dump(x2_train, open(to_save_path+"x2_train.pkl", 'wb'), protocol=4)
    pickle.dump(d_train, open(to_save_path+"d_train.pkl", 'wb'), protocol=4)
    pickle.dump(y_train, open(to_save_path+"y_train.pkl", 'wb'), protocol=4)
    pickle.dump(x1_val, open(to_save_path+"x1_val.pkl", 'wb'), protocol=4)
    pickle.dump(x2_val, open(to_save_path+"x2_val.pkl", 'wb'), protocol=4)

    pickle.dump(dataset, open(to_save_path+"dataset.pkl", 'wb'), protocol=4)
    pickle.dump(scalar, open(to_save_path+"scalar.pkl", 'wb'), protocol=4)

    final_preds_at = stitch_preds(preds_at_train, preds_at_val, past_history, past_history+future_target * STEP, int((future_target * STEP)*2))
    model_attention_GRU = tf.keras.models.load_model(to_save_path+'model_attention_GRU', custom_objects={'WeightedAverage': WeightedAverage})

    pickle.dump(final_preds_at, open(to_save_path+"final_preds_at.pkl", 'wb'), protocol=4)

# In[7]:
def model_predicting(master_df, location_name, up_stream_runoff_prediction_list,
        Output_path, past_history, future_target, STEP = 1, EPOCHS = 50,
        BATCH_SIZE = 512, LR = 0.0001, CELLS= 512, model_name_prefix = ""):

    #Scalar = StandardScaler()
    #scalar_filename = "scalar_m_"+model_name_prefix+".save"
    #joblib.dump(Scalar, scalar_filename)


    data = master_df[['p_'+str(location_name),'s_'+str(location_name)]]
    to_save_path = Output_path+ str(location_name)+ "_" + model_name_prefix + "/"+ model_name_prefix + "_"

    scalar = pickle.load(open(to_save_path+"scalar.pkl", 'rb'))


    if len(up_stream_runoff_prediction_list) == 0:
        data['up_'+str(location_name)] = 0.0
    else:
        for inx, v in enumerate(up_stream_runoff_prediction_list):
            data['up_'+str(location_name)+"_"+str(inx+1)] = v
            cols1 = data.columns.tolist()
            cols1 = cols1[-1:] + cols1[:-1]
            data = data[cols1]
    dataset= scalar.transform(data.values)

    TRAIN_SPLIT = int((4*len(master_df))/5)

    TRAIN_SPLIT = 0

    future_target = future_target * STEP
    #future_target = int(futures[0]) * STEP


    x1_val, x2_val, d_val, y_val = multivariate_data(dataset, dataset[:, -1],
                                                 TRAIN_SPLIT, None, past_history,
                                                 future_target, STEP)



    #opt =  tf.keras.optimizers.RMSprop(lr=0.001)

#     model_attention_GRU = tf.keras.models.load_model(to_save_path+'model_attention_GRU.h5', custom_objects={'WeightedAverage': WeightedAverage})
    model_attention_GRU = tf.keras.models.load_model(to_save_path+'model_attention_GRU', custom_objects={'WeightedAverage': WeightedAverage})

    preds_at_val =  getpreds_general(len(x1_val), model_attention_GRU, [x1_val, x2_val],'all');
    preds_at_val = Interpolate(preds_at_val, past_history, future_target)
    #Saving important data
    with open(to_save_path+'predictions.pkl', 'wb') as f:
        pickle.dump(preds_at_val, f)

    #dataset = pickle.load( open( to_save_path+"dataset.pkl", "rb" ))

    if len(up_stream_runoff_prediction_list) <2:
        df_at2 = get_df(dataset, scalar, preds_at_val, location_name)
    else:
        df_at2 = get_df2(dataset, scalar, preds_at_val, location_name)

    pickle.dump(df_at2, open(to_save_path+"_"+"df_attention.pkl", 'wb'), protocol=4)




def main(model_train,master_df_path,location_name, up_stream_runoff_prediction_list,
         Output_path,  past_history, future_target, STEP, EPOCHS,
         BATCH_SIZE, LR, CELLS, model_name_prefix,
         **hparams):

    master_df = pd.read_csv(master_df_path)
    if model_train == 1:
        model_training(master_df, location_name, up_stream_runoff_prediction_list,
        Output_path,  past_history, future_target, STEP, EPOCHS,
        BATCH_SIZE, LR, CELLS, model_name_prefix)
    else:
        model_predicting(master_df, location_name, up_stream_runoff_prediction_list,
        Output_path,  past_history, future_target, STEP, EPOCHS,
        BATCH_SIZE, LR, CELLS, model_name_prefix)





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--model_train',
      type=int,
      required=True,
      default=1,
      help='1 for training 0 for prediction')
    parser.add_argument(
      '--master_df_path',
      type=str,
      required=True,
      default=os.getcwd()+ "/masterdf.csv",
      help='Path to the input master df')

    parser.add_argument(
      '--location_name',
      type=str,
      required=True,
      default="",
      help='Current location name/id')
    parser.add_argument(
      "--up_stream_runoff_prediction_list",  # name on the CLI - drop the `--` for positional/required parameters
      nargs="*",  # 0 or more values expected => creates a list
      type=str,
      default=[], # default if nothing is provided)
      help='The list of all upstream location IDs. On commandline give them using spaces between the IDS. Example --up_stream_runoff_prediction_list 434478 399711')

    parser.add_argument(
      '--Output_path',
      type=str,
      required=True,
      default=os.getcwd()+ "/",
      help='path to store the output')
    parser.add_argument(
      '--past_history',
      type=int,
      default=96,
      help='Number of hours in the past to take as an input')
    parser.add_argument(
      '--future_target',
      type=int,
      default=48,
      help='Number of hours in the future to predict')
    parser.add_argument(
      '--STEP',
      type=int,
      default=1,
      help="step size")
    parser.add_argument(
      '--EPOCHS',
      type=int,
      default=50,
      help='Number of epochs')
    parser.add_argument(
      '--BATCH_SIZE',
      type=int,
      default=512,
      help='Batch size for training.')
    parser.add_argument(
      '--LR',
      type=float,
      default=0.0001,
      help="""\
      This is the inital learning rate value. The learning rate will decrease
      during training. For more details check the model_fn implementation in
      this file.\
      """)
    parser.add_argument(
      '--CELLS',
      type=int,
      default=512,
      help='Number of cells in each RNN layer')
    parser.add_argument(
      '--model_name_prefix',
      type=str,
      required=True,
      default="Model",
      help="""\
      model name prefix to store the trained model or model name reffered to load the model while prediction\
      """)
    args = parser.parse_args()

    main(**vars(args))
