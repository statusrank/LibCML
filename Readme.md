# A libray of Collaborative Metric Learning (CML)
This is an easy-to-use pytorch library of Collaborative Metric Learning (CML) based recommendation algorithms.

## Brief Introduction

In this repo, we implement some state-of-the-art CML-based methods, where the concept of CML was proposed in 2017 by [C.K Hsieh et al.](https://www.cs.cornell.edu/~ylongqi/paper/HsiehYCLBE17.pdf) and then sparked broad research interests in the community.

Overall, we have included the following promising approaches:
- [Collaborative Metric Learning](https://www.cs.cornell.edu/~ylongqi/paper/HsiehYCLBE17.pdf) (**CML**, WWW, 2017): A competitive model, which bridges the effort of metric learning and collaborative filtering.

- [Collaborative Translational Metric Learning](https://arxiv.org/abs/1906.01637) (**TransCF**, ICDM, 2018): A translation-based method. Specifically, such translation-based algorithms employ $d(i,j)=\|u_i + r_{ij} - v_j\|$ as the distance/score that discovers such user–item translation vectors via the users’ relationships with their neighbor items.

- [Latent Relational Metric Learning](https://arxiv.org/pdf/1707.05176.pdf) (**LRML**, WWW, 2018): Also a translation-based CML method. As a whole, the key idea of LRML is similar to TransCF. The main difference is how to access the translation vectors effectively. Concretely, TransCF leverages the neighborhood information of users and items to acquire the translation vectors while LRML introduces an attention-based memory augmented neural architecture to learn the exclusive and optimal translation vectors.

- [Adaptive Collaborative Metric Learning](https://link.springer.com/chapter/10.1007/978-3-030-18579-4_18)(**AdaCML**, DASFAA, 2019):

- [Co-occurrence embedding Regularized Metric Learning](https://dl.acm.org/doi/abs/10.1016/j.neunet.2020.01.021) (**CRML**, Neural Networks, 2020) considers the global statistical information of user-user and item-item pairs by involving a co-occurrence embedding to regularize the metric learning model. Then, CRML regards the optimization
problem as a multi-task learning problem to boost the performance of CML, including the primary CML recommendation task and two auxiliary representation learning tasks.

- [Hierarchical Collaborative Metric Learning](https://arxiv.org/abs/2108.04655)(**HLR**, RecSys, 2021):

- [Collaborative Preference Embedding](http://www.jdl.link/doc/2011/20191229_ACMM-Collaborative%20Preference%20Embedding%20against%20Sparse%20Labels.pdf) (**CPE**, ACM MM, 2019): A novel collaborative metric learning to effectively address the problem of sparse and insufficient preference supervision from the **margin distribution** point-of-view.

- [Diversity-Promoting Collaborative Metric Learning](https://arxiv.org/pdf/2209.15292.pdf) (**DPCML**, NeurIPS, 2022): A state-of-art Diversity-Promoting Collaborative Metric Learning (DPCML), with the hope of considering the commonly ignored minority interest of the user.

- [Sampling-Free Collaborative Metric Learning](https://arxiv.org/pdf/2206.11549.pdf) (**SFCML**, TPAMI, 2022): A state-of-art algorithm to learn CML without negative sampling to get rid of the bias, from which an effective acceleration method is constructed to overcome the heavy computational burden. 

# Citation
Please cite our paper if you find this libray is helpful.

> @inproceedings{DPCML, 
author    = {Shilong Bao, Qianqian Xu, Zhiyong Yang, Yuan He, Xiaochun Cao and Qingming Huang},
  title     = {The Minority Matters: A Diversity-Promoting Collaborative Metric Learning Algorithm},
  booktitle = {NeurIPS},
  year      = {2022}
}

> @article{DBLP:journals/pami/BaoX0CH23,
  author       = {Shilong Bao and
                  Qianqian Xu and
                  Zhiyong Yang and
                  Xiaochun Cao and
                  Qingming Huang},
  title        = {Rethinking Collaborative Metric Learning: Toward an Efficient Alternative
                  Without Negative Sampling},
  journal      = {{IEEE} Trans. Pattern Anal. Mach. Intell.},
  volume       = {45},
  number       = {1},
  pages        = {1017--1035},
  year         = {2023}
}

**Please feel relaxed to contact me at baoshilong@iie.ac.cn if there are any questions.**
