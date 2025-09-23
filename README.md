# MiN: Mixture of Noise for Pre-Trained Model-Based Class-Incremental Learning (NeurIPS 2025)

<p align="center">
  <a href='https://arxiv.org/abs/2509.16738'><img src='https://img.shields.io/badge/Arxiv-2509.16738-b31b1b.svg?logo=arXiv'></a>
  <!-- <a href=""><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fsun-hailong%2FLAMDA-PILOT&count_bg=%23FFA500&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false"></a> -->
  <a href=""><img src="https://black.readthedocs.io/en/stable/_static/license.svg"></a>
</p>


![Overall](source/overall_00.png)

# How to reproduce our method?

## Data
Six datasets are included in our experiments, i.e., CIFAR-100, CUB-200, ImageNet-A, ImageNet-R, FOOD-101 and Omnibenchmark.
1. **CIFAR-100**: it will be automatically downloaded by the code.
2. **ImageNet-A**: official: [link](https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar)
3. **Omnibenchmark**: Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EcoUATKl24JFo3jBMnTV2WcBwkuyBH0TmCAy6Lml1gOHJA?e=eCNcoA)
4. **CUB-200**: AWS: [link](https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz)
5. **FOOD-101**: AWS: [link](https://s3.amazonaws.com/fast-ai-imageclas/food-101.tgz)
6. **ImageNet-R**: official: [link](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar)
Then, revise the data path in data_process/data.py.

## Install
The core dependencies are as follows:
1. **torch==2.0.0+cu118** or **torch==2.8.0+cu12.8**
2. **torchvision==0.15.1+cu118** or **torchvision==0.23.0+cu118**
3. **timm**
4. **scikit-learn**
5. **scipy**

## How to run?
Use the script "run.sh" to reproduce the restlts of 10 steps setting:

**"sh run.sh"**

## Log Files
We hasv tested our code before uploading. The log files can be found in MiN/log. Due to the equipment change, we have to use 5090 to reproduce our results with torch==2.8.0+cu12.8. The result is basically the same as that stated in the paper.

## Main results

**CIFAR ViT-B/16-IN21K:**

![CIFAR](source/cifar100_in21k_00.png)

**CUB ViT-B/16-IN21K:**

![CUB](source/cub_in21k_00.png)

**IN-A ViT-B/16-IN21K:**

![IN-A](source/ina_in21k_00.png)

**IN-R ViT-B/16-IN21K:**

![IN-R](source/inr_in21k_00.png)

**FOOD ViT-B/16-IN21K:**

![IN-R](source/food_in21k_00.png)

**Omni. ViT-B/16-IN21K:**

![IN-R](source/omni_in21k_00.png)

**If you use any content of this repo for your work, please cite the following bib entries:**

    @inproceedings{MiN,
      title={Min: Mixture of Noise for Pre-Trained Model-Based Class-Incremental Learning}, 
      author={Kai Jiang and Zhengyan Shi and Dell Zhang and Hongyuan Zhang and Xuelong Li},
      booktitle={The Thirty-Ninth Annual Conference on Neural Information Processing Systems (NeurIPS)},
      year={2025},
      pages={},
    }
