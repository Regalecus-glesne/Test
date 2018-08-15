# -*- coding: utf-8 -*-

import numpy as np

class neuralNetwork:
    
    #初期化
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        #行列 wih who
        self.wih = np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        
        #学習率
        self.lr = learningrate
        
        #活性化関数:シグモイド関数,ソフトマックス関数
        self.activation_function_sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.activation_function_softmax = lambda x: np.exp(x) / np.sum(np.exp(x))
        
        pass
    
    #学習
    def train(self,inputs_list,targets_list):
        #行列変換
        inputs = np.array(inputs_list,ndmin=2).T
        targets = np.array(targets_list,ndmin=2).T
        
        #隠れ層の信号の計算
        hidden_inputs = np.dot(self.wih,inputs)
        #隠れ層の信号をシグモイド関数に通す
        hidden_outputs = self.activation_function_sigmoid(hidden_inputs)
        
        #出力層の信号の計算
        final_inputs = np.dot(self.who,hidden_outputs)
        #出力層の信号をソフトマックス関数に通す
        final_outputs = self.activation_function_softmax(final_inputs)
        
        #損失関数
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T,output_errors)
        #隠れ層と出力層間の重み更新
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        #入力層と隠れ層間の重み更新
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        
        pass
    
    #NNへの照会
    def query(self,inputs_list):
        #行列変換
        inputs = np.array(inputs_list,ndmin=2).T
        
        #隠れ層の信号の計算
        hidden_inputs = np.dot(self.wih,inputs)
        #隠れ層の信号を活性化(sig)
        hidden_outputs = self.activation_function_sigmoid(hidden_inputs)
        
        #出力層の信号の計算
        final_outputs = np.dot(self.who,hidden_outputs)
        
        return final_outputs