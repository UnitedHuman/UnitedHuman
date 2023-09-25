# UnitedHuman: Harnessing Multi-Source Data for High-Resolution Human Generation
<img src="./assets/teaser.png" width="96%" height="96%">

[Jianglin Fu](https://github.com/arleneF), [Shikai Li](mailto:lishikai@pjlab.org.cn), [Yuming Jiang](https://yumingj.github.io/), [Kwan-Yee Lin](https://kwanyeelin.github.io/), [Wayne Wu](https://wywu.github.io/), and [Ziwei Liu](https://liuziwei7.github.io/) <br>
**[[Video Demo]](https://www.youtube.com/watch?v=pdsfUYFDLSw)** | **[[Project Page]](https://unitedhuman.github.io/)** | **[[Paper]]()**

**Abstract:** *Human generation has achieved significant progress. Nonetheless, existing methods still struggle to synthesize specific regions such as faces and hands. We argue that the main reason is rooted in the training data. A holistic human dataset inevitably has insufficient and low-resolution information on local parts. Therefore, we propose to use multi-source datasets with various resolution images to jointly learn a high-resolution human generative model. 
However, multi-source data inherently a) contains different parts that do not spatially align into a coherent human, and b) comes with different scales. 
To tackle these challenges, we propose an end-to-end framework, UnitedHuman, that empowers continuous GAN with the ability to effectively utilize multi-source data for high-resolution human generation. Specifically, 1) we design a Multi-Source Spatial Transformer that spatially aligns multi-source images to full-body space with a human parametric model. 2) Next, a continuous GAN is proposed with global-structural guidance and CutMix consistency. Patches from different datasets are then sampled and transformed to supervise the training of this scale-invariant generative model. Extensive experiments demonstrate that our model jointly learned from multi-source data achieves superior quality than those learned from a holistic dataset.* <br>

## Updates
[13/07/2023] Our work has been accepted by ICCV2023!


## TODOs
- [ ] Release technical report.
- [ ] Release code.
- [ ] Release pretrained models.

## Related Work
* (ECCV 2022) **StyleGAN-Human: A Data-Centric Odyssey of Human Generation**, Jianglin Fu et al. [[Paper](https://arxiv.org/pdf/2204.11823.pdf)], [[Project Page](https://stylegan-human.github.io/)], [[Dataset](https://github.com/stylegan-human/StyleGAN-Human)]

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{0000,
    title={}, 
    author={Fu, Jianglin and Li, Shikai and Jiang, Yuming and Lin, Kwan-Yee and Wu, Wayne and Liu, Ziwei},
    journal   = {arXiv preprint},
    volume    = {arXiv:0000.0000},
    year    = {2023}
}
```
