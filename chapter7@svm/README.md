### 使用序列最小最优（SMO）算法训练支持向量机。
SMO是支持向量机学习的一种快速算法，其特点是不断地将原二次规划问题（QP）分解为只有两个变量的二次规划子问题，并对子问题进行解析求解，直到所有变量满足KKT条件为止，这样通过启发式的方法得到原QP问题最优解。因为子问题有解析解，子问题计算速度很快，SMO是一种高效算法。  
参考原始SMO算法论文 [Fast Training of Support Vector Machines Using Sequential Minimal Optimization](https://www.microsoft.com/en-us/research/publication/fast-training-of-support-vector-machines-using-sequential-minimal-optimization/)

#### result
![svmshow.png](https://raw.githubusercontent.com/fishermanff/lihang-book-impl/master/chapter7@svm/svmshow.png)