# -*- coding: utf-8 -*-
"""
0~9までの文字を認識する
結果が表示されるので、TrueかFalseを入力する
*logはans_nn_tr.txtを参照
"""
#NN用
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image
#import imageio
#クラス引用
from neuralNetwork import neuralNetwork
#dumpロード
import pickle

#nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

#学習率=0.1
learning_rate = 0.1

#nnインスタンス作成
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

#学習データ読み込み
with open('ihdata.pickle','rb') as ih:
    n.wih = pickle.load(ih)
with open('hodata.pickle','rb') as ho:
    n.who = pickle.load(ho)

#判定データロードしデジタル化
input_data_array = scipy.misc.imread("E:/手書き文字_mnist/mnist/input/input_data_8.png",flatten = True)
#input_data_array = imageio.imread("G:/手書き文字_mnist/mnist/input/input_data_3.png")
input_data = 255.0 - input_data_array.reshape(784)
input_data_cl = (np.asfarray(input_data) / 255.0 * 0.99) + 0.01

#nnのテスト

#実行
inputs = input_data_cl
#nn照会
outputs = n.query(inputs)
#解答のラベルを対応させる
label = np.argmax(outputs)
#nnの解答を表示
print("答え= ",label)
img = Image.open("E:/手書き文字_mnist/mnist/input/input_data_8.png")
img_list = np.asarray(img)
plt.imshow(img_list)
plt.show()
ans_end = input("True/False? ")
#結果を書き込む
ans_nn_tr = open("E:/手書き文字_mnist/mnist/output/ans_nn_tr.txt","a")
ans_nn_tr_list = [" 正誤= ",str(ans_end)]
ans_nn_tr.writelines(ans_nn_tr_list)
ans_nn_tr.close()
pass