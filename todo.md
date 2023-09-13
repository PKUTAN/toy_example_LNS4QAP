### 需要完成的任务

* [ ] 如何将gurobi用在更大的qap problem上：variable在80个以上的都不好求解，e.g. tai80，tai100
* [ ] 在进行local search的时候，对于sub problem的repair上，如何让objective function仅仅计算repair_size的计算量
* [ ] 如何将强化学习引入到local search当中
* [ ] 研究和实验是否可以加入adpative acceptance criteria来提升搜索的效率？

### 一些 insight

* 对于local search，比较关键在于：
  1、正确的匹配节点应该选择固定。
  2、local search点的个数不应该过大，10个节点一组比较合适。

### 一些实验的效果：

1. 对于tai64这类大的qap，LNS w/o learning 可以将gap缩小至 5%，这比直接用Gurobi（无法求解），或者RGM（ICLR2022）都要效果好很多。

### 一些问题：

1. LNS 如何实现adaptive acceptance criteria？
