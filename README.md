# curated-document-layout-analysis
This repository provides a curated list of resources in the research domain of Document Layout Analysis (DLA), updated in my free time.

Methods based on heuristic rules are not included.

## Keywords
"Document Layout Analysis" OR "Document Layout Detection" OR "Page Object Detection" OR "Document Structure Analysis" OR "Document Structure Extraction" OR "Document Reading Order" OR "Document Hierarchy"

## Table of Contents
1. Survey / Competition
2. Methodology
   - 2.1 Page Object Detection
   - 2.2 Document Structure Analysis
3. Data
   - 3.1 Dataset
   - 3.2 Data Synthesis / Data Generation / Data Augmentation

## 1. Survey / Competition
- [Datasets and annotations for layout analysis of scientific articles](https://link.springer.com/article/10.1007/s10032-024-00461-2), (UNIFI), 2024-03, IJDAR-2024
- [Advancements in Financial Document Structure Extraction: Insights from Five Years of FinTOC (2019-2023)](https://ieeexplore.ieee.org/document/10386125), (Outscale, NITK, Lancaster), 2023-12, BigData-2023
- [Document AI: A Comparative Study of Transformer-Based, Graph-Based Models, and Convolutional Neural Networks For Document Layout Analysis](https://arxiv.org/abs/2308.15517), (UVA, Elsevier), 2023-08, arXiv
- [ICDAR 2023 Competition on Robust Layout Segmentation in Corporate Documents](https://arxiv.org/abs/2305.14962), (IBM), 2023-05, ICDAR-2023
- [A Methodological Study of Document Layout Analysis](https://ieeexplore.ieee.org/document/10062927), (XJU), 2022-10, VRHCIAI-2022
- [A Survey of Graphical Page Object Detection with Deep Neural Networks](https://www.mdpi.com/2076-3417/11/12/5344), (TU Kaiserslautern, DFKI), 2021-06, Applied-Sciences.2021
- [ICDAR 2021 Competition on Scientific Literature Parsing](https://arxiv.org/abs/2106.14616), (IBM, UniMelb, Oracle), 2021-06, ICDAR-2021
- [Document Layout Analysis: A Comprehensive Survey](https://dl.acm.org/doi/10.1145/3355610), (KFUPM), 2019-10, ACM-Computing-Surveys.2019
- [ICDAR2019 Competition on Recognition of Documents with Complex Layouts - RDCL2019](https://ieeexplore.ieee.org/document/8978185), (Salford), 2019-09, ICDAR-2019
- [ICDAR2017 Competition on Page Object Detection](https://ieeexplore.ieee.org/document/8270162), (PKU), 2017-11, ICDAR-2017
- [ICDAR2017 Competition on Recognition of Documents with Complex Layouts - RDCL2017](https://ieeexplore.ieee.org/document/8270160), (Salford), 2017-11, ICDAR-2017

## 2. Methodology

### 2.1 Page Object Detection
- `DREAM` [DREAM: Document Reconstruction via End-to-end Autoregressive Model](https://arxiv.org/abs/2507.05805), (Tencent), 2025-07, ACM-MM-2025
- `DocSAM` [DocSAM: Unified Document Image Segmentation via Query Decomposition and Heterogeneous Mixed Learning](https://arxiv.org/abs/2504.04085), (CAS, UCAS), 2025-04, CVPR-2025
- `DLAdapter` [SFDLA: Source-Free Document Layout Analysis](https://arxiv.org/abs/2503.18742), (KIT, ETH Zurich), 2025-03, arXiv
- `PP-DocLayout` [PP-DocLayout: A Unified Document Layout Detection Model to Accelerate Large-Scale Data Construction](https://arxiv.org/abs/2503.17213), (Baidu), 2025-03, arXiv
- `UniHDSA` [UniHDSA: A Unified Relation Prediction Approach for Hierarchical Document Structure Analysis](https://arxiv.org/abs/2503.15893), (USTC, Microsoft), 2025-03, Pattern-Recognition.2025
- `EDocNet` [EDocNet: Efficient Datasheet Layout Analysis Based on Focus and Global Knowledge Distillation](https://arxiv.org/abs/2502.16541), (SEU), 2025-02, arXiv
- `DoPTA` [DoPTA: Improving Document Layout Analysis using Patch-Text Alignment](https://arxiv.org/abs/2412.12902), (Adobe), 2024-12, arXiv
- `DocEDA` [DocEDA: Automated Extraction and Design of Analog Circuits from Documents with Large Language Model](https://arxiv.org/abs/2412.05301), (SEU), 2024-12, arXiv
- `DocLayout-YOLO` [DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data and Global-to-Local Adaptive Perception](https://arxiv.org/abs/2410.12628), (Shanghai AI Lab), 2024-10, arXiv
- `UnSupDLA` [UnSupDLA: Towards Unsupervised Document Layout Analysis](https://arxiv.org/abs/2406.06236), (TU Kaiserslautern, DFKI), 2024-06, ICDAR-2024-Workshop
- `DLAFormer` [DLAFormer: An End-to-End Transformer For Document Layout Analysis](https://arxiv.org/abs/2405.11757), (USTC, Microsoft), 2024-05, ICDAR-2024
- `CREPE` [CREPE: Coordinate-Aware End-to-End Document Parser](https://arxiv.org/abs/2405.00260), (Naver, LINE WORKS), 2024-05, ICDAR-2024
- [A Hybrid Approach for Document Layout Analysis in Document images](https://arxiv.org/abs/2404.17888), (TU Kaiserslautern, DFKI), 2024-04, ICDAR-2024
- `M2Doc` [M2Doc: A Multi-Modal Fusion Approach for Document Layout Analysis](https://ojs.aaai.org/index.php/AAAI/article/view/28552), (SCUT, Alibaba), 2024-03, AAAI-2024
- `RoDLA` [RoDLA: Benchmarking the Robustness of Document Layout Analysis Models](https://arxiv.org/abs/2403.14442), (KIT, Oxford), 2024-03, CVPR-2024
- `Detect-Order-Construct` [Detect-Order-Construct: A Tree Construction based Approach for Hierarchical Document Structure Analysis](https://arxiv.org/abs/2401.11874), (USTC, Microsoft), 2024-01, Pattern-Recognition.2024
- [The YOLO model that still excels in document layout analysis](https://link.springer.com/article/10.1007/s11760-023-02838-y), (XJU), 2023-11, Signal-Image-and-Video-Processing.2024
- `DOLNet` [Dataset agnostic document object detection](https://www.sciencedirect.com/science/article/pii/S0031320323003965), (IIIT), 2023-10, Pattern-Recognition.2023
- `DSG` [DSG: An End-to-End Document Structure Generator](https://arxiv.org/abs/2310.09118), (ETH Zurich, LMU Munich), 2023-10, ICDM-2023
- `VGT` [Vision Grid Transformer for Document Layout Analysis](https://arxiv.org/abs/2308.14978), (Alibaba), 2023-08, ICCV-2023
- [A Hybrid Approach to Document Layout Analysis for Heterogeneous Document Images](https://link.springer.com/chapter/10.1007/978-3-031-41734-4_12), (Microsoft, USTC, PKU), 2023-08, ICDAR-2023
- `GLAM` [A Graphical Approach to Document Layout Analysis](https://arxiv.org/abs/2308.02051), (Kensho, Meta, Google), 2023-08, ICDAR-2023
- `HiM` [HiM: hierarchical multimodal network for document layout analysis](https://link.springer.com/article/10.1007/s10489-023-04782-3), (QUST, ASU), 2023-07, Applied-Intelligence.2023
- `DRFN` [DRFN: A unified framework for complex document layout analysis](https://www.sciencedirect.com/science/article/abs/pii/S0306457323000766), (ECNU, Fudan, Videt), 2023-05, Information-Processing-&-Management.2023
- `TransDLANet` [M6Doc: A Large-Scale Multi-Format, Multi-Type, Multi-Layout, Multi-Language, Multi-Annotation Category Dataset for Modern Document Layout Analysis](https://arxiv.org/abs/2305.08719), (SCUT, Huawei, IntSig), 2023-05, CVPR-2023
- `WeLayout` [WeLayout: WeChat Layout Analysis System for the ICDAR 2023 Competition on Robust Layout Segmentation in Corporate Documents](https://arxiv.org/abs/2305.06553), (Tencent), 2023-05, arXiv
- `SwinDocSegmenter` [SwinDocSegmenter: An End-to-End Unified Domain Adaptive Transformer for Document Instance Segmentation](https://arxiv.org/abs/2305.04609), (UAB, ISI), 2023-05, ICDAR-2023
- `SelfDocSeg` [SelfDocSeg: A Self-Supervised vision-based Approach towards Document Segmentation](https://arxiv.org/abs/2305.00795), (ISI, UAB, IITK), 2023-05, ICDAR-2023
- `StrucTexTv2` [StrucTexTv2: Masked Visual-Textual Prediction for Document Image Pre-training](https://arxiv.org/abs/2303.00289), (Baidu), 2023-03, ICLR-2023
- `HSCA-Net` [HSCA-Net: A Hybrid Spatial-Channel Attention Network in Multiscale Feature Pyramid for Document Layout Analysis](https://ojs.istp-press.com/jait/article/view/145), (QUST, PITT), 2022-12, Journal-of-Artificial-Intelligence-and-Technology.2023
- [Rethinking Learnable Proposals for Graphical Object Detection in Scanned Document Images](https://www.mdpi.com/2076-3417/12/20/10578), (TU Kaiserslautern, DFKI, LUT), 2022-10, Applied-Sciences.2020
- `TRDLU` [Transformer-Based Approach for Document Layout Understanding](https://ieeexplore.ieee.org/document/9897491), (KSU), 2022-10, ICIP-2022
- `PP-StructureV2` [PP-StructureV2: A Stronger Document Analysis System](https://arxiv.org/abs/2210.05391), (Baidu), 2022-10, arXiv
- [Lateral Feature Enhancement Network for Page Object Detection](https://ieeexplore.ieee.org/document/9866812), (QUST), 2022-08, TIM.2022
- `Doc-GCN` [Doc-GCN: Heterogeneous Graph Convolutional Networks for Document Layout Analysis](https://arxiv.org/abs/2208.10970), (Sydney, UWA), 2022-08, COLING-2022
- [Investigating Attention Mechanism for Page Object Detection in Document Images](https://www.mdpi.com/2076-3417/12/15/7486), (TU Kaiserslautern, DFKI, LTU), 2022-07, Applied-Sciences.2022
- `DSAP` [Exploiting Spatial Attention and Contextual Information for Document Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-031-05981-0_21), (XMU, Northumbria, SZU), 2022-05, PAKDD-2022
- `LayoutLMv3` [LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking](https://arxiv.org/abs/2204.08387), (SYSU, Microsoft), 2022-04, ACM-MM-2022
- `SRRV` [SRRV: A Novel Document Object Detector Based on Spatial-Related Relation and Vision](https://ieeexplore.ieee.org/document/9751354), (QUST), 2022-04, IEEE-Transactions-on-Multimedia.2022
- `DiT` [DiT: Self-supervised Pre-training for Document Image Transformer](https://arxiv.org/abs/2203.02378), (SJTU, Microsoft), 2022-03, ACM-MM-2022
- `DocBed` [DocBed: A Multi-Stage OCR Solution for Documents with Complex Layouts](https://arxiv.org/abs/2202.01414), (Amazon), 2022-02, IAAI-2022
- `DocSegTr` [DocSegTr: An Instance-Level End-to-End Document Image Segmentation Transformer](https://arxiv.org/abs/2201.11438), (UAB, ISI), 2022-01, arXiv
- `UDoc` [Unified Pretraining Framework for Document Understanding](https://arxiv.org/abs/2204.10939), (Adobe), 2021-12, NeurIPS-2021
- `L-E3Net` [Document Layout Analysis with Aesthetic-Guided Image Augmentation](https://arxiv.org/abs/2111.13809), (Fudan, ECNU, Videt), 2021-11, arXiv
- [Deep Learning in Time-Frequency Domain for Document Layout Analysis](https://ieeexplore.ieee.org/document/9605682), (UDLA, DETRI, UNICAMP, Analog Devices, ESPE), 2021-11, IEEE-Access.2021
- `E3Net` [Document image layout analysis via explicit edge embedding network](https://www.sciencedirect.com/science/article/abs/pii/S0020025521007106), (ECNU, Videt), 2021-10, Information-Sciences.2021
- [A Page Object Detection Method Based on Mask R-CNN](https://ieeexplore.ieee.org/document/9583930), (QUST, ASU, HIT), 2021-10, IEEE-Access.2021
- `VTLayout` [VTLayout: Fusion of Visual and Text Features for Document Layout Analysis](https://arxiv.org/abs/2108.13297), (UCAS, CAS), 2021-08, PRICAI-2021
- `FS-PARN` [Few-shot prototype alignment regularization network for document image layout segementation](https://www.sciencedirect.com/science/article/pii/S0031320321000698), (UESTC, YZU, Kyudai, Afanti, QDU), 2021-07, Pattern-Recognition.2021
- [End-to-end dilated convolution network for document image semantic segmentation](https://link.springer.com/article/10.1007/s11771-021-4731-9), (QUST, ASU), 2021-07, Journal-of-Central-South-University.2021
- [Beyond document object detection: instance-level segmentation of complex layouts](https://link.springer.com/article/10.1007/s10032-021-00380-6), (UAB, ISI), 2021-07, IJDAR.2021
- `SelfDoc` [SelfDoc: Self-Supervised Document Representation Learning](https://arxiv.org/abs/2106.03331), (Brandeis, Adobe), 2021-06, CVPR-2021
- `VSR` [VSR: A Unified Framework for Document Layout Analysis combining Vision, Semantics and Relations](https://arxiv.org/abs/2105.06220), (Hikvision, ZJU), 2021-05, ICDAR-2021
- [Document Layout Analysis with an Enhanced Object Detector](https://ieeexplore.ieee.org/document/9483509), (TU Kaiserslautern, DFKI), 2021-04, IPRIA-2021
- `LAMPRET` [LAMPRET: Layout-Aware Multimodal PreTraining for Document Understanding](https://arxiv.org/abs/2104.08405), (UCLA, Google), 2021-04, arXiv
- `DRFN` [Document Layout Analysis via Dynamic Residual Feature Fusion](https://arxiv.org/abs/2104.02874), (ECNU, Videt), 2021-04, ICME-2021
- [Probabilistic homogeneity for document image segmentation](https://www.sciencedirect.com/science/article/pii/S0031320320303940), (VUB), 2021-01, Pattern-Recognition.2021
- `BINYAS` [BINYAS: a complex document layout analysis system](https://link.springer.com/article/10.1007/s11042-020-09832-3), (GKCIET, Jadavpur), 2020-11, Multimedia-Tools-and-Applications.2021
- [Vision-Based Layout Detection from Scientific Literature using Recurrent Convolutional Neural Networks](https://arxiv.org/abs/2010.11727), (KSU), 2020-10, ICPR-2020
- [Page Segmentation Using Convolutional Neural Network and Graphical Model](https://link.springer.com/chapter/10.1007/978-3-030-57058-3_17), (CAS, UCAS), 2020-08, DAS-2020
- [Document Structure Extraction using Prior based High Resolution Hierarchical Semantic Segmentation](https://arxiv.org/abs/1911.12170), (Adobe), 2019-11, ECCV-2020
- `DocParser` [DocParser: Hierarchical Structure Parsing of Document Renderings](https://arxiv.org/abs/1911.01702), (ETH Zurich), 2019-11, AAAI-2021
- [Instance Aware Document Image Segmentation using Label Pyramid Networks and Deep Watershed Transformation](https://ieeexplore.ieee.org/document/8978152), (CAS, UCAS, Tencent, univ-lr), 2019-09, ICDAR-2019
- [Page Segmentation using a Convolutional Neural Network with Trainable Co-Occurrence Features](https://ieeexplore.ieee.org/document/8978118), (Kyudai, SIT), 2019-09, ICDAR-2019
- [Page Object Detection from PDF Document Images by Deep Structured Prediction and Supervised Clustering](https://ieeexplore.ieee.org/document/8546073), (CAS, UCAS), 2018-08, ICPR-2018
- `DeepLayout` [DeepLayout: A Semantic Segmentation Approach to Page Layout Analysis](https://link.springer.com/chapter/10.1007/978-3-319-95957-3_30), (PKU), 2018-07, ICIC-2018
- `dhSegment` [dhSegment: A generic deep-learning approach for document segmentation](https://arxiv.org/abs/1804.10371), (EPFL), 2018-04, ICFHR-2018
- [Ensemble of Deep Object Detectors for Page Object Detection](https://dl.acm.org/doi/10.1145/3164541.3164644), (VNU-HCM), 2018-01, IMCOM-2018
- [Multi-Scale Multi-Task FCN for Semantic Page Segmentation and Table Detection](https://ieeexplore.ieee.org/document/8269981), (PSU, Adobe), 2017-11, ICDAR-2017
- [CNN Based Page Object Detection in Document Images](https://ieeexplore.ieee.org/document/8269977), (PKU), 2017-11, ICDAR-2017
- [Fast CNN-Based Document Layout Analysis](https://ieeexplore.ieee.org/document/8265351), (IBM), 2017-10, ICCV-Workshop-2017
- `MFCN` [Learning to Extract Semantic Structure from Documents Using Multimodal Fully Convolutional Neural Network](https://arxiv.org/abs/1706.02337), (PSU, Adobe), 2017-06, CVPR-2017

### 2.2 Document Structure Analysis
- `DREAM` [DREAM: Document Reconstruction via End-to-end Autoregressive Model](https://arxiv.org/abs/2507.05805), (Tencent), 2025-07, ACM-MM-2025
- `SCAN` [SCAN: Semantic Document Layout Analysis for Textual and Visual Retrieval-Augmented Generation](https://arxiv.org/abs/2505.14381), (NEC), 2025-05, arXiv
- `XY-Cut++` [XY-Cut++: Advanced Layout Ordering via Hierarchical Mask Mechanism on a Novel Benchmark](https://arxiv.org/abs/2504.10258), (TJU), 2025-04, arXiv
- `UniHDSA` [UniHDSA: A Unified Relation Prediction Approach for Hierarchical Document Structure Analysis](https://arxiv.org/abs/2503.15893), (USTC, Microsoft), 2025-03, Pattern-Recognition.2025
- `DRGG` [Graph-based Document Structure Analysis](https://arxiv.org/abs/2502.02501), (KIT), 2025-02, ICLR-2025
- `DLAFormer` [DLAFormer: An End-to-End Transformer For Document Layout Analysis](https://arxiv.org/abs/2405.11757), (USTC, Microsoft), 2024-05, ICDAR-2024
- [Leveraging Collection-Wide Similarities for Unsupervised Document Structure Extraction](https://arxiv.org/abs/2402.13906), (Allen AI, HUJI, BIU), 2024-02, ACL-2024
- `Detect-Order-Construct` [Detect-Order-Construct: A Tree Construction based Approach for Hierarchical Document Structure Analysis](https://arxiv.org/abs/2401.11874), (USTC, Microsoft), 2024-01, Pattern-Recognition.2024
- `DSG` [DSG: An End-to-End Document Structure Generator](https://arxiv.org/abs/2310.09118), (ETH Zurich, LMU Munich), 2023-10, ICDM-2023
- [A Hybrid Approach to Document Layout Analysis for Heterogeneous Document Images](https://link.springer.com/chapter/10.1007/978-3-031-41734-4_12), (Microsoft, USTC, PKU), 2023-08, ICDAR-2023
- [Text Reading Order in Uncontrolled Conditions by Sparse Graph Segmentation](https://arxiv.org/abs/2305.02577), (Google), 2023-05, ICDAR-2023
- `DSPS` [HRDoc: Dataset and Baseline Method Toward Hierarchical Reconstruction of Document Structures](https://arxiv.org/abs/2303.13839), (USTC, iFLYTEK), 2023-02, AAAI-2023
- `LayerDoc` [LayerDoc: Layer-wise Extraction of Spatial Hierarchical Structure in Visually-Rich Documents](https://ieeexplore.ieee.org/document/10030764), (UMD, Adobe), 2023-01, WACV-2023
- `MTD` [Multimodal Tree Decoder for Table of Contents Extraction in Document Images](https://arxiv.org/abs/2212.02896), (USTC, iFLYTEK), 2022-08, ICPR-2022
- `XYLayoutLM` [XYLayoutLM: Towards Layout-Aware Multimodal Networks For Visually-Rich Document Understanding](https://arxiv.org/abs/2203.06947), (SJTU, Ant Group), 2022-03, CVPR-2022
- [Reading order detection on handwritten documents](https://link.springer.com/article/10.1007/s00521-022-06948-5), (UPV), 2022-02, Neural-Computing-and-Applications.2022
- `LayoutReader` [LayoutReader: Pre-training of Text and Layout for Reading Order Detection](https://arxiv.org/abs/2108.11591), (UC San Diego, Microsoft), 2021-08, EMNLP-2021
- `HELD` [Extracting Variable-Depth Logical Document Hierarchy from Long Documents: Method, Evaluation, and Application](https://arxiv.org/abs/2105.09297), (CAS, UCAS, Tencent, Peng Cheng Lab), 2021-05, Journal-of-Computer-Science-and-Technology.2022
- `LAMPRET` [LAMPRET: Layout-Aware Multimodal PreTraining for Document Understanding](https://arxiv.org/abs/2104.08405), (UCLA, Google), 2021-04, arXiv
- [An End-to-End OCR Text Re-organization Sequence Learning for Rich-Text Detail Image Comprehension](https://link.springer.com/chapter/10.1007/978-3-030-58595-2_6), (ZJU, Alibaba), 2020-11, ECCV-2020
- `DocStruct` [DocStruct: A Multimodal Method to Extract Hierarchy Structure in Document for General Form Understanding](https://arxiv.org/abs/2010.11685), (Sensetime), 2020-10, EMNLP-2020
- `DocParser` [DocParser: Hierarchical Structure Parsing of Document Renderings](https://arxiv.org/abs/1911.01702), (ETH Zurich), 2019-11, AAAI-2021
- [Table-Of-Contents generation on contemporary documents](https://arxiv.org/abs/1911.08836), (Fortia), 2019-09, ICDAR-2019

## 3. Data

### 3.1 Dataset
- `DocRec1K` [DREAM: Document Reconstruction via End-to-end Autoregressive Model](https://arxiv.org/abs/2507.05805), (Tencent), 2025-07, ACM-MM-2025
- `DocBench-100` [XY-Cut++: Advanced Layout Ordering via Hierarchical Mask Mechanism on a Novel Benchmark](https://arxiv.org/abs/2504.10258), (TJU), 2025-04, arXiv
- `AnnoPage` [AnnoPage Dataset: Dataset of Non-Textual Elements in Documents with Fine-Grained Categorization](https://arxiv.org/abs/2503.22526), (VUTBR, MZK, NKP, CAS), 2025-03, arXiv
- `SFDLA` [SFDLA: Source-Free Document Layout Analysis](https://arxiv.org/abs/2503.18742), (KIT, ETH Zurich), 2025-03, arXiv
- `GraphDoc` [Graph-based Document Structure Analysis](https://arxiv.org/abs/2502.02501), (KIT), 2025-02, ICLR-2025
- `OmniDocBench` [OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations](https://arxiv.org/abs/2412.07626), (Shanghai AI Lab, Abaka AI, 2077AI), 2024-12, CVPR-2025
- `LADaS 2.0` [Diachronic Document Dataset for Semantic Layout Analysis](https://arxiv.org/abs/2411.10068), (Inria, PSL, IHEID, UNIGE), 2024-11, arXiv
- `READoc` [READoc: A Unified Benchmark for Realistic Document Structured Extraction](https://arxiv.org/abs/2409.05137), (CAS, UCAS), 2024-09, arXiv
- `SciPostLayout` [SciPostLayout: A Dataset for Layout Analysis and Layout Generation of Scientific Posters](https://arxiv.org/abs/2407.19787), (OMRON SINIC X, Waseda), 2024-07, BMVC-2024
- `Comp-HRDoc` [Detect-Order-Construct: A Tree Construction based Approach for Hierarchical Document Structure Analysis](https://arxiv.org/abs/2401.11874), (USTC, Microsoft), 2024-01, Pattern-Recognition.2024
- `ADOPD` [ADOPD: A Large-Scale Document Page Decomposition Dataset](https://openreview.net/forum?id=x1ptaXpOYa), (Adobe, OSU, UC Merced, JHU), 2024-01, ICLR-2024
- `E-Periodica` [DSG: An End-to-End Document Structure Generator](https://arxiv.org/abs/2310.09118), (ETH Zurich, LMU Munich), 2023-10, ICDM-2023
- `D4LA` [Vision Grid Transformer for Document Layout Analysis](https://arxiv.org/abs/2308.14978), (Alibaba), 2023-08, ICCV-2023
- `CDSSE` [DRFN: A unified framework for complex document layout analysis](https://www.sciencedirect.com/science/article/abs/pii/S0306457323000766), (ECNU, Fudan, Videt), 2023-05, Information-Processing-&-Management.2023
- `M6Doc` [M6Doc: A Large-Scale Multi-Format, Multi-Type, Multi-Layout, Multi-Language, Multi-Annotation Category Dataset for Modern Document Layout Analysis](https://arxiv.org/abs/2305.08719), (SCUT, Huawei, IntSig), 2023-05, CVPR-2023
- `ETD-ODv2` [A New Annotation Method and Dataset for Layout Analysis of Long Documents](https://dl.acm.org/doi/10.1145/3543873.3587609), (Virginia Tech), 2023-04, WWW-2023
- `BaDLAD` [BaDLAD: A Large Multi-Domain Bengali Document Layout Analysis Dataset](https://arxiv.org/abs/2303.05325), (Bengali.AI, SUST, BRACU, Vanderbilt, Rice, RPI), 2023-03, ICDAR-2023
- `HRDoc` [HRDoc: Dataset and Baseline Method Toward Hierarchical Reconstruction of Document Structures](https://arxiv.org/abs/2303.13839), (USTC, iFLYTEK), 2023-02, AAAI-2023
- `TexBiG` [A Dataset for Analysing Complex Document Layouts in the Digital Humanities and Its Evaluation with Krippendorff’s Alpha](https://link.springer.com/chapter/10.1007/978-3-031-16788-1_22), (uni-weimar), 2022-09, DAGM-GCPR-2022
- `HierDoc` [Multimodal Tree Decoder for Table of Contents Extraction in Document Images](https://arxiv.org/abs/2212.02896), (USTC, iFLYTEK), 2022-08, ICPR-2022
- `DocLayNet` [DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis](https://arxiv.org/abs/2206.01062), (IBM), 2022-06, SIGKDD-2022
- `NewsNet7` [DocBed: A Multi-Stage OCR Solution for Documents with Complex Layouts](https://arxiv.org/abs/2202.01414), (Amazon), 2022-02, IAAI-2022
- `DAD` [Segmentation for document layout analysis: not dead yet](https://link.springer.com/article/10.1007/s10032-021-00391-3), (USask, Living Sky), 2022-01, IJDAR.2022
- `FPD` [Document Layout Analysis with Aesthetic-Guided Image Augmentation](https://arxiv.org/abs/2111.13809), (Fudan, ECNU, Videt), 2021-11, arXiv
- `SciBank` [Deep Learning in Time-Frequency Domain for Document Layout Analysis](https://ieeexplore.ieee.org/document/9605682), (UDLA, DETRI, UNICAMP, Analog Devices, ESPE), 2021-11, IEEE-Access.2021
- `ReadingBank` [LayoutReader: Pre-training of Text and Layout for Reading Order Detection](https://arxiv.org/abs/2108.11591), (UC San Diego, Microsoft), 2021-08, EMNLP-2021
- `DocBank` [DocBank: A Benchmark Dataset for Document Layout Analysis](https://arxiv.org/abs/2006.01038), (BUAA, Microsoft), 2020-06, COLING-2020
- `arXivdocs` [DocParser: Hierarchical Structure Parsing of Document Renderings](https://arxiv.org/abs/1911.01702), (ETH Zurich), 2019-11, AAAI-2021
- `article-regions` [Visual Detection with Context for Document Layout Analysis](https://aclanthology.org/D19-1348/), (BNL), 2019-11, EMNLP-IJCNLP-2019
- `PubLayNet` [PubLayNet: largest dataset ever for document layout analysis](https://arxiv.org/abs/1908.07836), (IBM), 2019-08, ICDAR-2019
- `ICDAR2017_POD` [ICDAR2017 Competition on Page Object Detection](https://ieeexplore.ieee.org/document/8270162), (PKU), 2017-11, ICDAR-2017
- `DSSE-200` [Learning to Extract Semantic Structure from Documents Using Multimodal Fully Convolutional Neural Network](https://arxiv.org/abs/1706.02337), (PSU, Adobe), 2017-06, CVPR-2017

### 3.2 Data Synthesis / Data Generation / Data Augmentation
- `LED Benchmark` [LED Benchmark: Diagnosing Structural Layout Errors for Document Layout Analysis](https://arxiv.org/abs/2507.23295), (CNU), 2025-07, arXiv
- `DocSynth-300K` [DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data and Global-to-Local Adaptive Perception](https://arxiv.org/abs/2410.12628), (Shanghai AI Lab), 2024-10, arXiv
- [RoDLA: Benchmarking the Robustness of Document Layout Analysis Models](https://arxiv.org/abs/2403.14442), (KIT, Oxford), 2024-03, CVPR-2024
- `LACE` [Towards Aligned Layout Generation via Diffusion Model with Aesthetic Constraints](https://arxiv.org/abs/2402.04754), (Buffalo, Adobe, MBZUAI), 2024-02, ICLR-2024
- `RanLayNet` [RanLayNet: A Dataset for Document Layout Detection used for Domain Adaptation and Generalization](https://arxiv.org/abs/2404.09530), (IIIT, VIT, JNU, NII), 2024-01, MMAsia-2023
- [WeLayout: WeChat Layout Analysis System for the ICDAR 2023 Competition on Robust Layout Segmentation in Corporate Documents](https://arxiv.org/abs/2305.06553), (Tencent), 2023-05, arXiv
- [A New Annotation Method and Dataset for Layout Analysis of Long Documents](https://dl.acm.org/doi/10.1145/3543873.3587609), (Virginia Tech), 2023-04, WWW-2023
- [Automatic generation of scientific papers for data augmentation in document layout analysis](https://www.sciencedirect.com/science/article/pii/S0167865523000247), (UNIFI), 2023-03, Pattern-Recognition-Letters.2023
- `LayoutDiffusion` [LayoutDiffusion: Improving Graphic Layout Generation by Discrete Diffusion Probabilistic Models](https://arxiv.org/abs/2303.11589), (SJTU, Microsoft), 2023-03, ICCV-2023
- [Diffusion-based Document Layout Generation](https://arxiv.org/abs/2303.10787), (Purdue, Microsoft), 2023-03, ICDAR-2023
- `LayoutDM` [LayoutDM: Discrete Diffusion Model for Controllable Layout Generation](https://arxiv.org/abs/2303.08137), (CyberAgent, Waseda), 2023-03, CVPR-2023
- `LDGM` [Unifying Layout Generation with a Decoupled Diffusion Model](https://arxiv.org/abs/2303.05049), (XJTU, Microsoft, Tsinghua), 2023-03, CVPR-2023
- `PLay` [PLay: Parametrically Conditioned Layout Generation using Latent Diffusion](https://arxiv.org/abs/2301.11529), (Google), 2023-01, ICML-2023
- `LayoutFormer++` [LayoutFormer++: Conditional Graphic Layout Generation via Constraint Serialization and Decoding Space Restriction](https://arxiv.org/abs/2208.08037), (XJTU, Microsoft, SJTU, BUAA), 2022-08, CVPR-2023
- [Coarse-to-Fine Generative Modeling for Graphic Layouts](https://ojs.aaai.org/index.php/AAAI/article/view/19994), (XJTU, Microsoft), 2022-06, AAAI-2022
- `DL-DSG` [Cross-Domain Document Layout Analysis Using Document Style Guide](https://arxiv.org/abs/2201.09407), (Fudan, ECNU, Videt), 2022-01, Expert-Systems-with-Applications.2024
- `BLT` [BLT: Bidirectional Layout Transformer for Controllable Layout Generation](https://arxiv.org/abs/2112.05112), (CMU, Google), 2021-12, ECCV-2022
- [Synthetic Document Generator for Annotation-free Layout Recognition](https://arxiv.org/abs/2111.06016), (JPMorgan), 2021-11, Pattern-Recognition.2022
- [Document image layout analysis via explicit edge embedding network](https://www.sciencedirect.com/science/article/abs/pii/S0020025521007106), (ECNU, Videt), 2021-10, Information-Sciences.2021
- [A Page Object Detection Method Based on Mask R-CNN](https://ieeexplore.ieee.org/document/9583930), (QUST, ASU, HIT), 2021-10, IEEE-Access.2021
- `CanvasEmb` [CanvasEmb: Learning Layout Representation with Large-scale Pre-training for Graphic Design](https://dl.acm.org/doi/10.1145/3474085.3475541), (NUS, Microsoft, Meituan), 2021-10, ACM-MM-2021
- `LayoutMCL` [Diverse Multimedia Layout Generation with Multi Choice Learning](https://arxiv.org/abs/2301.06629), (UNSW, CSIRO), 2021-10, ACM-MM-2021
- [Constrained Graphic Layout Generation via Latent Optimization](https://arxiv.org/abs/2108.00871), (Waseda, CyberAgent), 2021-08, ACM-MM-2021
- `CanvasVAE` [CanvasVAE: Learning to Generate Vector Graphic Documents](https://arxiv.org/abs/2108.01249), (CyberAgent), 2021-08, ICCV-2021
- `DocSynth` [DocSynth: A Layout Guided Approach for Controllable Document Image Synthesis](https://arxiv.org/abs/2107.02638), (UAB, ISI), 2021-07, ICDAR-2021
- `LayoutGAN` [LayoutGAN: Synthesizing Graphic Layouts With Vector-Wireframe Adversarial Networks](https://ieeexplore.ieee.org/document/8948239), (BIT, Adobe), 2021-07, IEEE-Transactions-on-Pattern-Analysis-and-Machine-Intelligence.2021
- `DDR` [Document Domain Randomization for Deep Learning Document Layout Extraction](https://arxiv.org/abs/2105.14931), (OSU, Uni Vie, Inria, Uni Stuttgart, Nottingham, ODU, PSU), 2021-05, ICDAR-2021
- `RUITE` [RUITE: Refining UI Layout Aesthetics Using Transformer Encoder](https://dl.acm.org/doi/10.1145/3397482.3450716), (RWTH), 2021-04, IUI-2021
- `VTN` [Variational Transformer Networks for Layout Generation](https://arxiv.org/abs/2104.02416), (Google, ETH Zurich, TUM), 2021-04, CVPR-2021
- `LayoutTransformer` [LayoutTransformer: Layout Generation and Completion with Self-attention](https://arxiv.org/abs/2006.14615), (UMD, UC San Diego, Amazon), 2020-06, ICCV-2021
- [Cross-Domain Document Object Detection: Benchmark Suite and Method](https://arxiv.org/abs/2003.13197), (NEU, Adobe), 2020-03, CVPR-2020
- `NDN` [Neural Design Network: Graphic Layout Generation with Constraints](https://arxiv.org/abs/1912.09421), (Google, UC Merced, Yonsei, Georgia Tech), 2019-12, ECCV-2020
- `READ` [READ: Recursive Autoencoders for Document Layout Generation](https://arxiv.org/abs/1909.00302), (SFU, TAU, Amazon, Cornell), 2019-09, CVPR-2020-Workshop
- [Content-aware generative modeling of graphic design layouts](https://dl.acm.org/doi/10.1145/3306346.3322971), (CityUHK), 2019-07, ACM-Transactions-on-Graphics.2019
- `LayoutGAN` [LayoutGAN: Generating Graphic Layouts with Wireframe Discriminators](https://arxiv.org/abs/1901.06767), (BIT, Adobe), 2019-01, ICLR-2019
- [Multi-Scale Multi-Task FCN for Semantic Page Segmentation and Table Detection](https://ieeexplore.ieee.org/document/8269981), (PSU, Adobe), 2017-11, ICDAR-2017
- [Learning to Extract Semantic Structure from Documents Using Multimodal Fully Convolutional Neural Network](https://arxiv.org/abs/1706.02337), (PSU, Adobe), 2017-06, CVPR-2017
