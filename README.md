# Image coding and contrast learning

 This project includes a VIT model for image coding and a contrast learning network structure for training the model to generate image coding and perform matching tasks. This README file provides information on how to use and configure these components

## Install

First, make sure you have Python and the following dependencies installed:
- PyTorch=1.9.0
- torchvision
- numpy
- PIL
- tqdm

## File structure
The file structure of the project is as follows：
- project_root/
    - vit_encode.py
    - net.py
    - train.py
    - config.py
    - ContarstLoss.py
    - test_result.py
    - vit_encode_model.py
    - vit_base_patch16_224_in21k.pth
	- data/
    - tarin_40encode/
        - 0/
             - test
             - train
             - train_random
        - 18/
             - test
             - ...
        - ...
    - ckpt/
        - pretrained_model.pth
    - results/
        - 0/
            - loss.png
            - top1.png
        - 18/
            - ...
        - ...
    - README.md
-   `vit_encode.py`: A script for encoding images using a VIT pre-trained model.
-   `net.py`: Contains code for contrasting learning network structures.
-   `train.py`: It is used to train the model and generate the corresponding encoding file and match the TOP1 results.
-   `ContarstLoss.py`:Contarst loss.
-	`tarin_40encode`:Generated by vit_encode.py
-	`results`:Generated by train.py
-	`ckpt`:Generated by train.py

## Usage method

Run the `vit_encode.py` file to encode the image. You can use the following command:
~~~
python vit_encode.py
~~~
To generate a different Angle of gait feature encoding, please modify the `rain_source_folder` and `test_source_folder` in this file to suit your dataset. We used the CASIA-B and OU-MVLP datasets to verify our method

CASIA-B download address: http://www.cbsr.ia.ac.cn/china/Gait%20Databases%20CH.asp
OU-MVLP download address: http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitMVLP.html
Run the `train.py` file to train the model and generate results matching TOP1. You can use the following command:
~~~
python train.py
~~~
Please download`vit_base_patch16_224_in21k.pth` in advance into the project root directory
You can modify `train_encode_path` and `train_rand_encode_path` in `train.py` to modify the number of combinations and angles you want to train, and the results will be saved in the `results` folder.
The specific matching details will be saved in the `pair_detail` floder.
## Contribution
If you have any questions, suggestions or would like to contribute to this project, please feel free to contact us or create a question.