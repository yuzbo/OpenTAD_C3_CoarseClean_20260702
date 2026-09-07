# GeoSparse-TAD 源码审查快照（2026-09-08）

当前模型源码固定为 `902fa05b5c64452ff1c94b82cabce801943d3484`；本分支另外整理了审查材料。
从 [审查入口](GEOSPARSE_REVIEW.zh.md) 开始，或直接复制 [Pro 严格审查 prompt](review/PRO_REVIEW_PROMPT.zh.md)。
当前只推进 seed 0，未完成的实验不作结果声明。`review/historical/` 是撤销代码或旧方案，不是当前实现。

以下保留官方 OpenTAD README；其中的发布状态和结果属于上游项目。

# OpenTAD: An Open-Source Temporal Action Detection Toolbox.

<p align="left">
<a href="https://arxiv.org/abs/2502.20361" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2502.20361-b31b1b.svg?style=flat" /></a>
<a href="https://github.com/sming256/opentad/blob/main/LICENSE" alt="license">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" /></a>
<a href="https://github.com/sming256/OpenTAD/issues" alt="docs">
    <img src="https://img.shields.io/github/issues-raw/sming256/OpenTAD?color=%23FF9600" /></a>
<a href="https://img.shields.io/github/stars/sming256/opentad" alt="arXiv">
    <img src="https://img.shields.io/github/stars/sming256/opentad" /></a>
</p>

OpenTAD is an open-source temporal action detection (TAD) toolbox based on PyTorch.


## 🥳 What's New

- **[2024/07/25]** 🔥 We rank 1st in the [Action Recognition](https://codalab.lisn.upsaclay.fr/competitions/776#results), [Action Detection](https://codalab.lisn.upsaclay.fr/competitions/707), and [Audio-Based Interaction Detection](https://codalab.lisn.upsaclay.fr/competitions/17921#results) tasks of the EPIC-KITCHENS-100 2024 Challenge, as well as 1st place in the [Moment Queries](https://eval.ai/web/challenges/challenge-page/1626/leaderboard/3913) task of the Ego4D 2024 Challenge! Code is released at [CausalTAD (arxiv'24)](configs/causaltad/).
- **[2024/07/07]** 🔥 We support [DyFADet (ECCV'24)](configs/dyfadet/). Thanks to the authors's effort!
- **[2024/06/14]** We release [version v0.3](docs/en/changelog.md), which brings many new features and improvements.
- **[2024/04/17]** We release the [AdaTAD (CVPR'24)](configs/adatad), which can achieve average mAP of 42.90% on ActivityNet and 77.07% on THUMOS14.

## 📖 Major Features

- **Support SoTA TAD methods with modular design.** We decompose the TAD pipeline into different components, and implement them in a modular way. This design makes it easy to implement new methods and reproduce existing methods.
- **Support multiple TAD datasets.** We support 9 TAD datasets, including ActivityNet-1.3, THUMOS-14, HACS, Ego4D-MQ, EPIC-Kitchens-100, FineAction, Multi-THUMOS, Charades, and EPIC-Sounds Detection datasets.
- **Support feature-based training and end-to-end training.** The feature-based training can easily be extended to end-to-end training with raw video input, and the video backbone can be easily replaced.
- **Release various pre-extracted features.** We release the feature extraction code, as well as many pre-extracted features on each dataset.

## 🌟 Model Zoo

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>One Stage</b>
      </td>
      <td>
        <b>Two Stage</b>
      </td>
      <td>
        <b>DETR</b>
      </td>
      <td>
        <b>End-to-End Training</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
            <li><a href="configs/actionformer">ActionFormer (ECCV'22)</a></li>
            <li><a href="configs/tridet">TriDet (CVPR'23)</a></li>
            <li><a href="configs/temporalmaxer">TemporalMaxer (arXiv'23)</a></li>
            <li><a href="configs/videomambasuite">VideoMambaSuite (arXiv'24)</a></li>
            <li><a href="configs/dyfadet">DyFADet (ECCV'24)</a></li>
            <li><a href="configs/causaltad">CausalTAD (arXiv'24)</a></li>
      </ul>
      </td>
      <td>
        <ul>
            <li><a href="configs/bmn">BMN (ICCV'19)</a></li>
            <li><a href="configs/gtad">GTAD (CVPR'20)</a></li>
            <li><a href="configs/tsi">TSI (ACCV'20)</a></li>
            <li><a href="configs/vsgn">VSGN (ICCV'21)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/tadtr">TadTR (TIP'22)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/afsd">AFSD (CVPR'21)</a></li>
          <li><a href="configs/tadtr">E2E-TAD (CVPR'22)</a></li>
          <li><a href="configs/etad">ETAD (CVPRW'23)</a></li>
          <li><a href="configs/re2tal">Re2TAL (CVPR'23)</a></li>
          <li><a href="configs/adatad">AdaTAD (CVPR'24)</a></li>
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

The detailed configs, results, and pretrained models of each method can be found in above folders.

## 🛠️ Installation

Please refer to [install.md](docs/en/install.md) for installation.

## 📝 Data Preparation

Please refer to [data.md](docs/en/data.md) for data preparation.

## 🚀 Usage

Please refer to [usage.md](docs/en/usage.md) for details of training and evaluation scripts.


## 📄 Updates
Please refer to [changelog.md](docs/en/changelog.md) for update details.


## 🤝 Roadmap

All the things that need to be done in the future is in [roadmap.md](docs/en/roadmap.md).


## 🖊️ Citation

**[Acknowledgement]** This repo is inspired by [OpenMMLab](https://github.com/open-mmlab) project, and we give our thanks to their contributors.

If you think this repo is helpful, please cite us:

```bibtex
@article{liu2025opentad,
  title={OpenTAD: A Unified Framework and Comprehensive Study of Temporal Action Detection},
  author={Liu, Shuming and Zhao, Chen and Zohra, Fatimah and Soldan, Mattia and Pardo, Alejandro and Xu, Mengmeng and Alssum, Lama and Ramazanova, Merey and Alcázar, Juan León and Cioppa, Anthony and Giancola, Silvio and Hinojosa, Carlos and Ghanem, Bernard},
  journal={arXiv preprint arXiv:2502.20361},
  year={2025}
}
```

If you have any questions, please contact: `shuming.liu@kaust.edu.sa`.
