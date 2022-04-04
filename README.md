# RL-DiVTS: Selecting a Diverse Set of Aesthetically-Pleasing and Representative Video Thumbnails Using Reinforcement Learning

## PyTorch Implementation of RL-DiVTS  
<div align="justify">

- From **"RL-DiVTS: Selecting a Diverse Set of Aesthetically-Pleasing and Representative Video Thumbnails Using Reinforcement Learning"**.
- Written by Evlampios Apostolidis, Georgios Balaouras, Vasileios Mezaris and Ioannis Patras.

## Main dependencies
Developed, checked and verified on an `Ubuntu 20.04.3` PC with an `NVIDIA RTX 2080Ti` GPU and an `i5-11600K` CPU. Main packages required:
|`Python` | `PyTorch` | `CUDA Version` | `cuDNN Version` | `TensorBoard` | `TensorFlow` | `NumPy` | `H5py`
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
3.8(.8) | 1.7.1 | 11.0 | 8005 | 2.4.0 | 2.4.1 | 1.20.2 | 2.10.0

## Data
<div align="justify">

Structured h5 files with the video features and annotations of the OVP and YouΤube datasets are available within the [data](data) folder. The GoogleNet features of the video frames were extracted by [Ke Zhang](https://github.com/kezhang-cs) and [Wei-Lun Chao](https://github.com/pujols) and the h5 files were obtained from [Kaiyang Zhou](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce). These files have the following structure:
<pre><code>/key
    /features                 2D-array with shape (n_steps, feature-dimension), feature vectors representing the content of the video frames; extracted from the pool5 layer of a GoogleNet trained on the ImageNet dataset
    /aesthetic_scores_mean    1D-array with shape (n_steps), scores representing the aesthetic quality of the video frames; computed as the softmax of the values in the final layer of a model of a <a href="https://github.com/bmezaris/fully_convolutional_networks" target="_blank">Fully Convolutional Network</a> trained on the AVA dataset
    /n_frames                 number of frames in original video
    /ssim_matrix              2D-array with shape (top-5 selected thumbs, n_frames), the structural similarity scores between each of the five most selected thumbnails by the human annotators (in order to support evaluation using 'Precision at 5') and the entire frame sequence; computed using the <a href="https://scikit-image.org/docs/dev/api/skimage.metrics.html#skimage.metrics.structural_similarity" target="_blank">structural_similarity function</a> of Python
    /top1_thumbnail_ids       the index of the most selected thumbnail by the human annotators (can be more than one if they exist more than one key-frames with the same ranking according to the number of selections made by the human annotators)
    /top3_thumbnail_ids       the indices of the three most selected thumbnails by the human annotators (can be more than three if they exist more than three key-frames with the same ranking according to the number of selections made by the human annotators)
</code></pre>
Original videos and annotations for each dataset are also available here: 
- <a href="https://sites.google.com/site/vsummsite/download" target="_blank"><img align="center" src="https://img.shields.io/badge/Datasets-OVP,%20YouTube-green"/></a>
</div>
 
 ## Configurations
<div align="justify">

Setup for the training process:
 - In [`data_loader.py`](model/data_loader.py#L19:L21), specify the path to the h5 file of the used dataset, and the path to the JSON file containing data about the utilized data splits.
 - In [`configs.py`](model/configs.py#L7), define the directory where the analysis results will be saved to. </div>
   
Arguments in [`configs.py`](model/configs.py): 
|Parameter name | Description | Default Value | Options
| :--- | :--- | :---: | :---:
`--mode` | Mode for the configuration. | 'train' | 'train', 'test'
`--verbose` | Print or not training messages. | 'false' | 'true', 'false'
`--video_type` | Used dataset for training the model. | 'OVP' | 'OVP', 'Youtube'
`--input_size` | Size of the input feature vectors. | 1024 | int > 0
`--hidden_size` |Number of features in the LSTM hidden state. | 512 | int > 0
`--num_layers` | Number of LSTM recurrent layers. | 2 | int > 0
`--n_episodes` | Number of training episodes per epoch. | 10 | int > 0
`--selected_thumbs` | Number of selected thumbnails. | 5 | int > 0
`--n_epochs` | Number of training epochs. | 200 | int > 0
`--batch_size` | Size of the training batch, 40 for 'OVP' and 32 for 'Youtube'. | 40 | 0 < int ≤ len(Dataset)
`--seed` | Chosen number for generating reproducible random numbers. | None | None, int
`--exp` | Experiment serial number. | 1000 | int
`--clip` | Gradient norm clipping parameter. | 5.0 | float 
`--lr` | Value of the adopted learning rate. | 1e-4 | float
`--split_index` | Index of the utilized data split. | 0 | 0 ≤ int ≤ 9

