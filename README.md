<div align="center">

# Boosting All-in-One Image Restoration via Self-Improved Privilege Learning

[Gang Wu](https://scholar.google.com/citations?user=JSqb7QIAAAAJ), [Junjun Jiang](http://homepage.hit.edu.cn/jiangjunjun),  [Kui Jiang](https://github.com/kuijiang94), and [Xianming Liu](http://homepage.hit.edu.cn/xmliu)

[AIIA Lab](https://aiialabhit.github.io/team/), Faculty of Computing, Harbin Institute of Technology, Harbin 150001, China.

</div>

</div>

<a href="https://www.imagehub.cc/image/WX20250523-142230%402x.IcXcf6"><img src="https://s1.imagehub.cc/images/2025/05/23/184c1207eb1c32d9d6950600adf3b8fb.png" alt="WX20250523 142230@2x" border="0"></a>
## Overview
Unified image restoration models for diverse and mixed degradations often suffer from unstable optimization dynamics and inter-task conflicts. This paper introduces Self-Improved Privilege Learning (SIPL), a novel paradigm that overcomes these limitations by innovatively extending the utility of privileged information (PI) beyond training into the inference stage. Unlike conventional Privilege Learning, where ground-truth-derived guidance is typically discarded after training, SIPL empowers the model to leverage its own preliminary outputs as pseudo-privileged signals for iterative self-refinement at test time. Central to SIPL is Proxy Fusion, a lightweight module incorporating a learnable Privileged Dictionary. During training, this dictionary distills essential high-frequency and structural priors from privileged feature representations. Critically, at inference, the same learned dictionary then interacts with features derived from the model's initial restoration, facilitating a self-correction loop. SIPL can be seamlessly integrated into various backbone architectures, offering substantial performance improvements with minimal computational overhead. Extensive experiments demonstrate that SIPL significantly advances the state-of-the-art on diverse all-in-one image restoration benchmarks. For instance, when integrated with the PromptIR model, SIPL achieves remarkable PSNR improvements of +4.58 dB on composite degradation tasks and +1.28dB on diverse five-task benchmarks, underscoring its effectiveness and broad applicability.

## Results

<a href="https://www.imagehub.cc/image/WX20250523-142456%402x.IcXHLJ"><img src="https://s1.imagehub.cc/images/2025/05/23/afa655a3d6f21aa95a9da81d6fb28de0.png" alt="WX20250523 142456@2x" border="0"></a>

<a href="https://www.imagehub.cc/image/WX20250523-142508%402x.IcXsW7"><img src="https://s1.imagehub.cc/images/2025/05/23/dc98a3a0dc44b4cbc626bed48b85e606.png" alt="WX20250523 142508@2x" border="0"></a>

<a href="https://www.imagehub.cc/image/WX20250523-142518%402x.IcXj5e"><img src="https://s1.imagehub.cc/images/2025/05/23/87840d792a32fe43822d257bc01d74c6.png" alt="WX20250523 142518@2x" border="0"></a>

<a href="https://www.imagehub.cc/image/WX20250523-142529%402x.IcXJ3Z"><img src="https://s1.imagehub.cc/images/2025/05/23/cf1f5db7be63f77b832e71156a11de65.png" alt="WX20250523 142529@2x" border="0"></a>
