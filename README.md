
# EATD-Corpus 抑郁症倾向音频分类实验
----
本仓库提供了一个统一的实验框架，旨在使用**线性探测 (Linear Probing)** 策略，对多种预训练音频模型在`EATD-Corpus`数据集上进行抑郁症倾向的三分类微调。
---
部分模型文件夹具有以下结构：
```
project/
├── Audio_Flamingo-2/
│   ├── AFCLap/
│   │   ├── get_feature_2dep.py
│   │   └── get_feature_3emo.py
│   └── Flamingo-Embedding/
│       ├── flamingo_2dep_classifier.py
│       └── flamingo_3emo_classifier.py
│── CLAP/
│   └── src/laion_clap/
│   │   ├── get_feature_2dep.py
│   │   └── get_feature_3emo.py
│   └── CLAP-Embedding/
│       ├── clap_2dep_classifier.py
│       └── clap_3emo_classifier.py
│
...
```
---

## 🚀 部署

### 1. 环境配置

首先，创建并激活Conda环境，然后运行```pip install -r requirement.txt```安装所需的核心依赖库。
**注意**: 每个实验子目录可能包含其特定的依赖文件(如`CLAP/`需要执行`pip install laoin-clap`) ，请根据backbone模型环境的需要进行安装。

### 2. 数据准备

**这是一个关键步骤。** `EATD-Corpus` 数据集需要被分别放置在名为 `dataset` 的目录中。并运行对应`.py`文件对数据集进行处理。若希望进行抑郁症二分类任务，请运行 `data_2dep.py`文件；若希望进行情绪三分类任务，请运行`data_3emo.py`文件。


### 3. 获取Embedding

以二分类任务为例，需要执行每个项目中的`get_feature_2dep.py`文件，以获取经对应模型处理后的Embedding。注意，对应模型的加载路径需要根据你的实际路径进行修改。

### 4. 训练线性分类头

以`Qwen`模型二分类任务为例，执行`qw2_2dep_classifier.py`文件以将所提取的Embedding作为输入，训练一个Linear分类器。

---
## 🚀 快速训练一个分类头

本项目已提供各个模型提取后的Embedding，因此在配置环境后执行对应的`classifier.py`文件即可训练和执行分类任务。

---

## 📊 性能表现

本项目采用**线性探测 (Linear Probing)** 微调策略。所有预训练骨干网络 (Backbone) 的参数均被冻结，仅训练一个在其之上新增的线性分类头。

各模型在 `EATD-Corpus` 执行情绪三分类任务上的类别准确率总结如下：


| 骨干模型 (Backbone) | **Negative** | **Neutral** | **Positive** |
|:---|:---:|:---:|:---:|
| **CLAP** | *0.5442* | *0.4354* | *0.5374* |
| **Qwen2-Audio** | *0.5986* | *0.5442* | *0.5102* |
| **Audio-Flamingo-2** | *0.5510* | *0.5306* | *0.5170* |

各模型在 `EATD-Corpus` 验证集执行二分类任务上的表现总结如下：

| 骨干模型 (Backbone) | 验证集准确率 (Accuracy) |
| :------------------ |:-----------------:|
| **CLAP**            |     *0.5156*      |
| **Qwen2-Audio**     |     *0.6518*      |
| **Audio-Flamingo-2**|     *0.6562*      |



## 📦 模型库 (Model Zoo)

本项目中所使用的模型及微调策略详情。

| 实验文件夹         | 基础检查点 (Base Checkpoint)                     | 微调策略 (Fine-tuning Strategy) |
| :------------------- |:--------------------------------------------|:----------------------------|
| `CLAP/`              | `laion/630k-audioset-fusion-best`           | 线性探测 (Linear Probing)       |
| `Qwen2_Audio/`       | `Qwen/Qwen2-Audio-7B`                       | 线性探测 (Linear Probing)       |
| `Audio_Flamingo_2/`  | `nvidia/audio-flamingo-2-0.5b（epoch_16.pt）` | 线性探测  (Linear Probing)      |
