DeepConvLSTM Implementation in Tensorflow based on Ordoñez et al. :
Ordóñez, F. J. and Roggen, D. (2016) Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable
Activity Recognition, Sensors, 16(1). doi: 10.3390/s16010115.

and further inspired by: D. van Kuppevelt, C. Meijer, F. Huber, A. van der Ploeg, S. Georgievska, V.T. van Hees. Mcfly:
Automated deep learning on time series. SoftwareX, Volume 12, 2020. doi: 10.1016/j.softx.2020.100548

**Before initializing the network, please make sure to format the input data as following:
number_of windows * samples_per_window * number_of_channels**
**The labels need to be one-hot encoded**

**The network needs to initialized as an object. An example can be found at the end of the network definition.**


Please cite the paper "Digging deeper: towards a better understanding of transfer learning for human activity recognition", when you are using this implementation in your scientific work. 

@inproceedings{hoelzemann2020digging,
  title={Digging deeper: towards a better understanding of transfer learning for human activity recognition},
  author={Hoelzemann, Alexander and Van Laerhoven, Kristof},
  booktitle={Proceedings of the 2020 International Symposium on Wearable Computers},
  pages={50--54},
  year={2020}
}

For further questions please email alexander.hoelzemann@uni-siegen.de

