### 需要完成的任务

* [X] 如何将gurobi用在更大的qap problem上：variable在80个以上的都不好求解，e.g. tai80，tai100 （利用GPT，但是结果不尽人意，感觉是一个很技巧的工程问题）实际上花时间的应该是presolve，后续矩阵计算还算可以（**150以上确实算不动了，系统直接kill了**）
* [X] 如何将强化学习引入到local search当中，local-size and local variable selections are both very important
* [ ] 强化学习是否可以有multi policies for destroy purpose?
* [X] 是否可以生成synthetic data 4 GM 用来训练我们的强化学习模型？
* [ ] 研究和实验是否可以加入adaptive acceptance criteria来提升搜索的效率？
* [X] 在不同的epoch采用不同大小的local size，比如前几个epoch用大的local size，后几个epoch用较小的local size。（效果不显著，认为对于qap问题，选择正确的匹配比local size大小要关键）
* [ ] 用NGM生成一个初始解。
* [X] 以单位阵为初始解，每次local search重新定义目标函数，已达到减少计算量的目的
* [ ] 如何让gurobi不执着于证明最优解而是直接返回值
* [ ] 可以在上一次local search集合中的补集中选择新的neighboor，或者设计一个超参数，按照一些比例从原集和补集选择index。
* [ ] local search 中做 local search（一开始的local search 较大但相对于原问题比较小，然后较大的local search子问题再用local search快速得到最优解）
* [ ] 当到达了local optimal的时候应该如何跳出这个local plain？

### 一些 insight

* 对于local search，比较关键在于：
  * 正确的匹配节点应该选择固定。
  * local search点的个数不应该过大，10个节点一组比较合适。
  * local search size和time limit互相限制，合适的local search size和time limit是关键。
  * 有很多redundant search， RL的作用就体现出来了，可以用RL来避开not valid search
  * LNS容易stuck在sub optimal 平面中,initial sol 很重要，当stuck住的时候，如何perturb？

### 一些实验的效果

1. 对于tai64这类大的qap，random LNS w/o learning 可以将gap缩小至 5%，这比直接用Gurobi（无法求解），或者RGM（ICLR2022）都要效果好很多。
2. lipa60b gurobi能在200s找到optimal 解，但是random LNS 仍在挣扎
3. 例如tai150b，LNS会stuck在sub optimal处，此时一个不同的destroy function会有很大作用，stuck的这种现象可以从RL中学出？

### 一些问题

1. LNS - RL for QAP 和 LNS - RL for IP 有什么不同吗？
2. 目前random LNS在tai256c上还不如18年的whale algorithm for TS

### Difficult instances:

easy : chr , esc, had, scr,

**bur**系列对random LNS不友好，需要specific searching strategy

**kra**的random LNS略好于gurobi，有提升空间

lipa60b gurobi可找到最优解

lipa70a gurobi无法找到最优解, random LNS效果也不佳

**lipa**系列的random LNS效果都不好，有用ml提升空间

**nug**系列感觉也是可以用于训练random LNS的数据集

**rou(20)** 数据集比较依赖search的质量，感觉也可以用于LNS

**sko 是很好的数据集for training， sko64 gurobi kill**

80及其以上的qap， presolve时间过长,导致每次optimize时间过长（sko81,need 300s for presolve)

ste36a local size 20, time 10 iterate 50 10170 (当然越大的local size会得到越好的结果) ；36b 对于gurobi相对简单

tai系列: 30a random search lack than gurobi; 80及其以上的instance确实没有办法用LNS，原因是gurobi presolve的时间太长（目标函数计算过于困难）

总结： bur， chr25系列，kra30系列，lipa60，70系列， nug27以上的例子，rou20，sko56-72，ste 36， tai30-64系列总计28个例子可用于LNS

### 进度：

#### 9.19:

* [X] 画出random search LNS与Gurobi在时间与gap之间关系的图标，首先在一个instance上进行代码调试。

#### 9.20：

* [X] 继续调试更多的instance, 将所有instance都过一遍(仅过到了sko 81)

#### 9.21:

* [X] 继续过instance，今天需要过的是ste，tai和tho。

#### 9.26:

* [X] 解决无法处理过大instance的问题。


#### 10.10

* [ ] 阅读DDPG的论文
* [ ] 需要给observation 加入 batch normalization
