# -*- coding: utf-8 -*-
"""
学習データ作成
"""
#NN用
import numpy as np
#クラス引用
from neuralNetwork import neuralNetwork
from timer import StopWatch
#dump作成
import pickle


#nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

#学習率=0.1

learning_rate = 0.1

#nnインスタンス作成
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

#訓練csvロードしリスト化
training_data_file = open("G:/手書き文字_mnist/mnist/input/mnist_train.csv","r")
training_data_list = training_data_file.readlines()
training_data_file.close()

#nn学習
epochs = input("エポック: ")
epochs = int(epochs)
#訓練時間計測開始
train_time = StopWatch()
train_time.start()
for e in range(epochs):
#全訓練データ実行
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs,targets)
    pass
pass
#訓練時間計測終了
train_time_end = train_time.stop()
print(train_time_end)
#dump化
target_data_ih = n.wih
target_data_ho = n.who
fname_ih = 'ihdata.pickle'
fname_ho = 'hodata.pickle'

with open(fname_ih,'wb') as ih:
    pickle.dump(target_data_ih,ih)
with open(fname_ho,'wb') as ho:
    pickle.dump(target_data_ho,ho)


pass