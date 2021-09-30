DeepConvLSTM Implementation in Tensorflow based on Ordoñez et al. :
Ordóñez, F. J. and Roggen, D. (2016) Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable
Activity Recognition, Sensors, 16(1). doi: 10.3390/s16010115.

and further inspired by: D. van Kuppevelt, C. Meijer, F. Huber, A. van der Ploeg, S. Georgievska, V.T. van Hees. Mcfly:
Automated deep learning on time series. SoftwareX, Volume 12, 2020. doi: 10.1016/j.softx.2020.100548

**Before initializing the network, please make sure to format the input data as following:
number_of windows * samples_per_window * number_of_channels**
**The labels need to be one-hot encoded**

The network needs to initialized as an object. An example can be found at the end of the network definition.

For further questions please email alexander.hoelzemann@uni-siegen.de

