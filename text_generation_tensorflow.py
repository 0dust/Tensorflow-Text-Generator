#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 14:22:45 2018

@author: himanshu
"""

import numpy as np
import tensorflow as tf
import random 
import argparse
import os
import time

class Config:
    def __init__(self,sess,lstm_cell_size,number_of_layers,learning_rate,input_size,output_size,time_steps):
        self.sess = sess
       #self.scope = scope
        self.lstm_cell_size = lstm_cell_size
        self.number_of_layers = number_of_layers
        self.learning_rate = tf.constant(learning_rate)
        self.input_size = input_size
        self.output_size = output_size   
        self.time_steps = time_steps
        
class Model:
    def __init__(self,config):
        self.scope = 'rnn'
        self.config = config
        self.input_placeholder = None
        self.labels_placeholder = None
        self.lstm_last_state = np.zeros((self.config.number_of_layers * 2 * self.config.lstm_cell_size,))
        def build(self):
            self.add_placeholders()
            self.pred = self.add_prediction_op()
            self.loss = self.add_loss_op(self.pred)
            self.train_op = self.add_training_op(self.loss)
        build(self)     
      
    def add_placeholders(self):
        
        self.input_placeholder = tf.placeholder(name = "input_placeholder",dtype = tf.float32,shape = (None,None,self.config.input_size))
        self.label_placeholder = tf.placeholder(name = "label_placeholder",dtype = tf.int32,shape = (None,None,self.config.output_size))
        self.lstm_initial_state = tf.placeholder(name = "lstm_ini_state",dtype = tf.float32,shape = (None,self.config.number_of_layers*2*self.config.lstm_cell_size))
        
    def create_feed_dict(self,input_batch,label_batch):
       lstm_initial_state = np.zeros((input_batch.shape[0],self.config.number_of_layers*2*self.config.lstm_cell_size))
      # print('entered in dictionary')
       feed_dict = {
                self.input_placeholder : input_batch,
                self.label_placeholder : label_batch,
                self.lstm_initial_state : lstm_initial_state
                }
       
      # print('exiting dictionary')         
       return feed_dict
    
        
    def add_prediction_op(self):
        #add training here
    
        x = self.input_placeholder #x is of shape (None,time_steps,input_size)
        self.y = self.label_placeholder
        lstm_cells = [tf.contrib.rnn.BasicLSTMCell(self.config.lstm_cell_size,
                                                        forget_bias = 1,
                                                        state_is_tuple = False) for i in range(self.config.number_of_layers)]    
        lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells,state_is_tuple = False)
        outputs,self.lstm_new_state = tf.nn.dynamic_rnn(lstm,x,initial_state = self.lstm_initial_state,dtype = tf.float32)
        
        W = tf.Variable(tf.random_normal((self.config.lstm_cell_size, self.config.output_size),stddev=0.01),name = "W")
        b = tf.Variable(tf.random_normal((self.config.output_size,), stddev=0.01 ),name = "b")
        
        outputs_reshaped = tf.reshape(outputs, [-1, self.config.lstm_cell_size])
        pred = tf.matmul(outputs_reshaped,W) + b
        
        outputs_shape = tf.shape(outputs)
        
        self.final_outputs = tf.reshape(tf.nn.softmax(pred),(outputs_shape[0], outputs_shape[1], self.config.output_size))
       
        self.y = tf.reshape(self.y, [-1, self.config.output_size])
#        print(self.label_placeholder.shape)
#        print(self.y.shape)
#        print(pred.shape)
        return pred
    def add_loss_op(self,pred):
        #add loss operation here
        loss = tf.nn.softmax_cross_entropy_with_logits(logits = pred,labels = self.y )
        return loss
    
    def add_training_op(self,loss):
        train_op = tf.train.RMSPropOptimizer(self.config.learning_rate,0.9).minimize(self.loss)
        return train_op
   
    
    def generate_proba_for_talking(self,x,initial_state_of_lstm_is_zero = True):
        if(initial_state_of_lstm_is_zero):
            initial_state = np.zeros((self.config.number_of_layers*2*self.config.lstm_cell_size,))
        else:
            initial_state = self.lstm_last_state
        out,next_lstm_state = self.config.sess.run([self.final_outputs,self.lstm_new_state],feed_dict={self.input_placeholder:[x],
                                            self.lstm_initial_state :[initial_state]})    
        self.lstm_last_state = next_lstm_state[0]
        return out[0][0]
    
    def train_on_batch(self,input_batch,label_batch):
        feed = self.create_feed_dict(input_batch,label_batch)
        _,loss_ = self.config.sess.run([self.train_op,self.loss],feed_dict = feed)
        return loss_

def text_to_char_array(data,vocab):
    data_char_array = np.zeros((len(data),len(vocab)))
    row = 0
    for char_ in list(data):
        data_char_array[row,vocab.index(char_)] = 1
        row+=1
    return data_char_array    

def char_array_to_text(array,vocab):
    return vocab[array.index(1)]    

def create_train_data(input_file):  
    data = ""
    with open(input_file,'r') as f:
        data = data + f.read()
    data = data.lower()
    vocab = sorted(list(set(data)))
    train_data = text_to_char_array(data,vocab)
    return train_data,vocab

def check_restore(sess,saver):
    pickle_ = tf.train.get_checkpoint_state(os.path.dirname('saved/checkpoint'))
    if(pickle_ and pickle_.model_checkpoint_path):
        saver.restore(sess,pickle_.model_checkpoint_path)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",
                         type = str,
                         default = "shakespeare.txt",
                         help = "Text file used for creating training data")
    parser.add_argument("--test_prefix",
                         type = str,
                         default = "The ",
                         help = "Test text prefix to train the network")
    parser.add_argument("--ckpt_file",
                         type=str,
                         default="saved/model.ckpt",
                         help="Model checkpoint file to load.")
    parser.add_argument("--mode",
                         type = str,
                         default = "talk",
                         choices = set(("train","talk")),
                         help = "Execution mode: train or talk to trained model"
                         )
    args = parser.parse_args()

    pickle_file = None
    
    if(args.ckpt_file):
        pickle_file = args.ckpt_file
    
    train_data,vocab = create_train_data(args.input_file)
    
    input_size = output_size  = len(vocab)
    lstm_cell_size = 128
    number_of_layers = 2
    batch_size =128
    time_steps = 50
    learning_rate = 0.03
    test_prefix = args.test_prefix
    num_train_batches = 10000
    len_test_text = 500
    
    #tf.reset_default_graph()
    sess = tf.InteractiveSession()
    config = Config(sess,lstm_cell_size,number_of_layers,learning_rate,input_size,output_size,time_steps)
    
    model = Model(config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    
    if(args.mode == "train"):
        check_restore(sess,saver)
        last_time = time.time()
        batch_x = np.zeros((batch_size,time_steps,input_size))
        batch_y = np.zeros((batch_size,time_steps,input_size))
        
        possible_batch_ids = range(train_data.shape[0]-time_steps-1)
        
        for i in range(num_train_batches):
            batch_id = random.sample(possible_batch_ids,batch_size)
            
            for j in range(time_steps):
                ind1 = [k+j for k in batch_id]
                ind2 = [k+j+1 for k in batch_id]
                batch_x[:,j,:] = train_data[ind1,:]
                batch_y[:,j,:] = train_data[ind2,:]
               # print(batch_x.shape,batch_y.shape)
            cost = model.train_on_batch(batch_x,batch_y)    
            
            if(i%100 == 0):
                new_time = time.time()
                diff = new_time  - last_time
                last_time = new_time
                print("batch:{} loss: {} time_taken :{}".format(i,cost,diff))
                saver.save(sess,pickle_file)
                
    elif(args.mode == "talk"):
        saver.restore(sess,pickle_file)
        test_prefix = test_prefix.lower()
         
        for i in range(len(test_prefix)):
            out = model.generate_proba_for_talking(text_to_char_array(test_prefix[i],vocab),i==0)
        print("Sentence :")
        gen_str = test_prefix
        for i in range(len_test_text):
            element = np.random.choice(range(len(vocab)),p = out)
            gen_str = gen_str + vocab[element]
            out = model.generate_proba_for_talking(text_to_char_array(vocab[element],vocab),False)
        print(gen_str)
         
if(__name__ == "__main__"):
    main()                   
        
    
        
        
    
          
        
        

         
        
