# 다음 파일을 수정할 것

1. onlyAcc.py  
Accelerometer data만 50,000번 학습시키는 모델
---
1. onlyAud.py  
STFT audio data만 15,000번 학습시키는 모델
---
1. modelAdd.py  
STFT audio data로 6,000번 학습 후, 
Accelerometer data로 20,000번 학습시키는 모델 
`keras.layers.Add()([y_conv1, y_conv2])`

1. modelMax.py  
STFT audio data로 6,000번 학습 후, 
Accelerometer data로 20,000번 학습시키는 모델 
`tf.maximum(y_conv1, y_conv2)`

1. win17000drop.py  
Accelerometer data와 STFT audio data를 동시에 학습시키는 모델(15,000번 학습) 
Fully connected layer가 3개 있다. 
`keras.layers.Concatenate()([h_fc11, h_pool22_flat])`

1. win17000dropFc1.py  
Accelerometer data와 STFT audio data를 동시에 학습시키는 모델(15,000번 학습) 
Fully connected layer가 1개 있다. 
`keras.layers.Concatenate()([h_fc11, h_pool22_flat])`
