# 审稿意见整理

**稿件编号**: FGCS-D-26-02280
**题目**: Optimal Online Orchestration of Scalable Federated Learning for Massive Internet-of-Things
**期刊**: Future Generation Computer Systems
**修订截止日期**: 2026年8月14日

---

## Reviewer #1 — 主要意见 (Main Comments)

### Comment 1.3

**英文原文**:
The contribution to federated learning per se is limited. The optimization objective in problem (6) is the time-averaged admitted data volume, not any FL-specific convergence metric. There is no theoretical statement linking the throughput-optimal admission and placement decisions to FL model convergence, generalization, or fairness under non-IID data. Maximizing admitted training throughput and maximizing FL model quality are distinct objectives, and the former can lead to biased data concentration on a small set of workers, which is known to harm FL accuracy under non-IID distributions. The authors should either extend the framework to incorporate an FL-aware objective (for example, gradient-variance-aware or fairness-aware worker selection) and prove the corresponding optimality result, or reposition the contribution as throughput-optimal data-plane orchestration that supports FL training, rather than as an FL algorithm.

**中文翻译**:
对联邦学习本身的贡献有限。问题(6)中的优化目标是时间平均的接纳数据量，而非任何FL特定的收敛指标。没有理论陈述将吞吐量最优的接纳和放置决策与FL模型收敛性、泛化能力或非IID数据下的公平性联系起来。最大化接纳训练吞吐量与最大化FL模型质量是不同的目标，前者可能导致数据偏向集中于少数工作者集合，这在非IID分布下已知会损害FL准确率。作者应要么扩展框架以纳入FL感知的目标（例如，梯度方差感知或公平性感知的工作者选择）并证明相应的最优性结果，要么将贡献重新定位为支持FL训练的吞吐量最优数据平面编排，而非一种FL算法。

---

### Comment 1.4

**英文原文**:
The baselines used for the 9.3× throughput claim are insufficient. The comparison includes only "Local Training" (random worker selection without offloading) and "Random Orchestration" (random worker selection with offloading via Algorithm 1). Both baselines lack any form of optimized worker selection, which makes a large multiplicative gap unsurprising and difficult to interpret. The manuscript explicitly references FL client selection schemes [22], [23], [24] in Section 2 but does not compare against them. At least one published FL client selection method (for example, FedCS, Oort, Power-of-Choice, or one of [22]-[24]) and one greedy capacity-based heuristic should be included. The phrase "state of the art" in the abstract should not be used unless a comparison against a current state-of-the-art method is performed.

**中文翻译**:
用于支撑9.3倍吞吐量声称的基线方法不够充分。对比仅包括"Local Training"（无卸载的随机工作者选择）和"Random Orchestration"（通过算法1进行卸载的随机工作者选择）。两种基线均缺乏任何形式的工作者选择优化，这使得大幅倍数差距并不令人惊讶且难以解读。手稿在第2节中明确引用了FL客户端选择方案[22]、[23]、[24]，但未与之比较。应至少包含一种已发表的FL客户端选择方法（例如FedCS、Oort、Power-of-Choice，或[22]-[24]之一）和一种基于容量的贪心启发式方法。摘要中的"state of the art"表述不应使用，除非与当前最先进方法进行了比较。

### Comment 1.6

**英文原文**:
The bound in Theorem 2, namely εB with B = 2T(Σᵢθᵢᵐᵃˣδᵢᵐᵃˣ + Σ\_(i,j)Cᵢⱼᵐᵃˣθᵢⱼᵐᵃˣδᵢᵐᵃˣ), grows linearly with T, |N|, and |E| and depends on the maxima of stochastic parameters. In massive IoT settings with large N and dense E this constant can be substantial. The manuscript states that the loss "asymptotically diminishes as ε → 0" but does not quantify εB for the experimental setting ε = 1/2000. A numerical estimate of the bound under the actual simulation parameters should be provided, and the relationship between the bound and the observed gap between T=10 and T=20 configurations should be discussed.

**中文翻译**:
定理2中的界，即εB，其中B = 2T(Σᵢθᵢᵐᵃˣδᵢᵐᵃˣ + Σ\_(i,j)Cᵢⱼᵐᵃˣθᵢⱼᵐᵃˣδᵢᵐᵃˣ)，随T、|N|和|E|线性增长，且依赖于随机参数的最大值。在大规模IoT设置中，N较大且E密集，该常数可能相当可观。手稿声称该损失"随着ε → 0渐近消失"，但未对实验设置ε = 1/2000下的εB进行量化。应提供实际仿真参数下的界的数值估计，并讨论该界与T=10和T=20配置之间观察到的差距之间的关系。

---

### Comment 1.7

**英文原文**:
The reported CIFAR-10 accuracy values in Fig. 8 fall within 62%-68% after 100 FL rounds. VGG11 with FedAvg on CIFAR-10 typically reaches substantially higher accuracy at convergence even under moderate non-IID partitioning. The current accuracy regime suggests that training has not converged, which makes accuracy differences of approximately 1.89% between T=10 and T=20 difficult to interpret as a real effect rather than transient noise. The authors should either run the FL experiments for a larger number of rounds (sufficient for convergence under each configuration) or justify why 100 rounds is the appropriate operating point. The learning rate, batch size, aggregation rule, and any client sampling strategy used in the FL experiment should also be reviewed for consistency with standard FL baselines.

**中文翻译**:
图8中报告的CIFAR-10准确率值在100轮FL后落在62%-68%范围内。VGG11配合FedAvg在CIFAR-10上，即使在中度非IID划分下，收敛时通常也能达到显著更高的准确率。当前的准确率区间表明训练尚未收敛，这使得T=10与T=20之间约1.89%的准确率差异难以被解释为真实效果而非暂态噪声。作者应要么运行更多轮次的FL实验（足以使每种配置收敛），要么论证为什么100轮是合适的操作点。FL实验中使用的学习率、批次大小、聚合规则以及任何客户端采样策略也应进行审查，以与标准FL基线保持一致。

---

### Comment 1.8

**英文原文**:
The experimental results lack measures of statistical dispersion. Figures 4, 5, 6, 7, 8, and 9 report single point estimates without standard deviations, confidence intervals, or error bars. Given the stochastic nature of wireless channels, energy arrivals, and non-IID data sampling, multiple independent runs and dispersion statistics are required to support quantitative comparisons such as the 9.3× throughput gain or the 1.89% accuracy improvement. The number of independent runs per data point and the seed-handling procedure should be reported.

**中文翻译**:
实验结果缺乏统计离散度的度量。图4、5、6、7、8和9均报告了单点估计值，没有标准差、置信区间或误差条。鉴于无线信道、能量到达以及非IID数据采样的随机性，需要多次独立运行和离散度统计来支持定量比较，例如9.3倍吞吐量增益或1.89%准确率提升。应报告每个数据点的独立运行次数以及种子处理过程。

---

### Comment 1.9

**英文原文**:
The computational complexity and runtime of the overall framework are not analyzed. Algorithm 1 invokes Bellman-Ford on the residual graph at each augmenting iteration, and BPSO repeatedly invokes Algorithm 1 to evaluate v(s). The total per-frame complexity, the wall-clock time on a representative platform, and how these scale with N should be reported. Since the paper claims scalability to massive IoT, measurements at the larger end of the device-count sweep in Fig. 5(a) (N=200) are particularly relevant.

**中文翻译**:
整体框架的计算复杂度和运行时间未进行分析。算法1在每次增广迭代中调用残差图上的Bellman-Ford算法，而BPSO反复调用算法1以评估v(s)。应报告每帧的总复杂度、在代表性平台上的实际运行时间，以及这些指标随N的扩展情况。由于论文声称可扩展至大规模IoT，图5(a)中设备数扫描较大端（N=200）的测量数据尤为相关。

---

### Comment 1.10

**英文原文**:
The hyperparameters and convergence behavior of the BPSO component are not specified. The swarm size M, the parameters α and β in (19a), the maximum number of iterations, the stopping criterion, and the sensitivity of the final placement to these choices are not reported. BPSO does not provide a global-optimum guarantee for mixed-integer problems, so the term "optimal" applied to the overall scheme should be qualified.

**中文翻译**:
BPSO组件的超参数和收敛行为未予说明。未报告群体大小M、公式(19a)中的参数α和β、最大迭代次数、停止准则，以及最终放置对这些选择的敏感性。BPSO不对混合整数问题提供全局最优保证，因此应用于整体方案的"optimal"一词应加以限定。

---

### Comment 1.11

**英文原文**:
The result that T=20 yields higher convergence accuracy than T=10 (Fig. 8) runs counter to the common expectation that more frequent worker rotation increases data diversity at the parameter server and helps under non-IID conditions. The proposed explanation, based on reduced "model jitter," is plausible but not analyzed. The authors should report results for additional values of T (for example T=5, T=30, T=50), identify whether an optimum exists, and connect the observation to known FL phenomena such as stale aggregation or client drift.

**中文翻译**:
T=20比T=10产生更高收敛准确率的结果（图8）与常见预期相悖——更频繁的工作者轮换通常会增加参数服务器端的数据多样性，并在非IID条件下有所帮助。基于减少"模型抖动"的所提解释虽有一定道理但未被分析。作者应报告更多T值下的结果（例如T=5、T=30、T=50），确定是否存在最优值，并将该观察与已知的FL现象（如陈旧聚合或客户端漂移）联系起来。

---

### Comment 1.12

**英文原文**:
The i.i.d. assumption on network dynamics (Section 5.1) is restrictive. Wireless channels are typically temporally correlated due to shadowing and Doppler effects, energy harvesting often exhibits diurnal patterns, and IoT data arrivals frequently show periodic structure. The authors should either provide a robustness experiment with temporally correlated dynamics (for example a Markov-modulated process or trace-driven data) or discuss the conditions under which the asymptotic optimality result continues to hold.

**中文翻译**:
对网络动态的独立同分布假设（第5.1节）是受限的。无线信道通常因阴影效应和多普勒效应而存在时间相关性，能量收集通常呈现昼夜模式，IoT数据到达也经常表现出周期性结构。作者应要么提供具有时间相关动态的鲁棒性实验（例如马尔可夫调制过程或基于实测数据），要么讨论在何种条件下渐近最优性结果依然成立。

---

### Comment 1.13

**英文原文**:
The "Cost" metric used in Fig. 9 is not defined precisely in the main text. The units, the components that enter the cost (energy in mJ, weighted communication cost, or something else), and the procedure for accumulating cost until 55% accuracy is reached should be stated explicitly so that the reported reductions (for example 22.03% from H=5 to H=20) can be interpreted.

**中文翻译**:
图9中使用的"Cost"指标在正文中未精确定义。应明确说明单位、构成成本的组成部分（以mJ为单位的能耗、加权通信成本或其他），以及累积成本直至达到55%准确率的过程，以便能够理解所报告的降幅（例如从H=5到H=20的22.03%）。

---

### Comment 1.14

**英文原文**:
The relationship between Algorithm 1 and BPSO in the large-timescale step should be made more explicit. It should be clarified whether v(s) in (18b) is evaluated for every particle at every BPSO iteration, whether warm-starting is used, and how the inner-loop result interacts with the outer Lagrange-multiplier update across slots.

**中文翻译**:
应更明确地说明大时间尺度步骤中算法1与BPSO之间的关系。应澄清(18b)中的v(s)是否在每次BPSO迭代中对每个粒子都进行评估，是否使用了热启动，以及内环结果如何与跨时隙的外层拉格朗日乘子更新相互作用。

---

## Reviewer #1 — 次要意见 (Minor Comments)

### Comment 1.20

**英文原文**:
Figures 8 and 9 use 3D surface plots that are difficult to read. 2D heatmaps, or selected line plots showing accuracy versus α for each H, would communicate the results more clearly.

**中文翻译**:
图8和图9使用了难以阅读的3D曲面图。2D热力图，或显示每个H下准确率随α变化的选定折线图，将更清晰地传达结果。

### Comment 1.22

**英文原文**:
The version of VGG11 used in the experiments should be specified, in particular whether batch normalization is included, since this affects training behavior on CIFAR-10.

**中文翻译**:
实验中使用的VGG11版本应予说明，特别是是否包含批量归一化，因为这会影响在CIFAR-10上的训练行为。
