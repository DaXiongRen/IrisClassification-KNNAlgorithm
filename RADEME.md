# KNN算法实现鸢尾花数据集分类 C语言实现(附数据集)



# KNN算法介绍

KNN的全称是K Nearest Neighbors，意思是K个最近的邻居，从这个名字我们就能看出一些KNN算法的蛛丝马迹了。K个最近邻居，毫无疑问，K的取值肯定是至关重要的。那么最近的邻居又是怎么回事呢？其实啊，KNN的原理就是当预测一个新的值x的时候，根据它距离最近的K个点是什么类别来判断x属于哪个类别。听起来有点绕，还是看看图吧。
![原理图](https://img-blog.csdnimg.cn/img_convert/de5bf66bd84ae88aa9c096796ba021e4.png)
图中绿色的点就是我们要预测的那个点，假设K=3。那么KNN算法就会找到与它距离最近的三个点（这里用圆圈把它圈起来了），看看哪种类别多一些，比如这个例子中是蓝色三角形多一些，新来的绿色点就归类到蓝三角了。

# 欧几里得距离介绍
## 定义
欧几里得距离（ Euclidean distance）也称欧式距离，它是一个通常采用的距离定义，它是在m维空间中两个点之间的真实距离。
## 公式
$$
二维：d=\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}
$$
$$
n维：d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
$$
我们用到n维的

# 实现思路

## 数据集

特征值的类别数：即花萼长度、花萼宽度、花瓣长度、花瓣宽度。
三种鸢尾花：setosa、versicolor、virginica。
(部分)
![部分数据集截图](https://img-blog.csdnimg.cn/3f4924e466a948598fc90ace8fa24311.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5aSn54aK5Lq6,size_20,color_FFFFFF,t_70,g_se,x_16)



## 实现步骤

① 读取数据，打乱数据（或者随机读取数据），并把每种花分别设置**A、B、C**标签。

② 分割数据（共150组，分55组为测试集，95组为训练集）。

③遍历$K(1\leq K \leq 15,K\%2\neq0)$值。

④ 计算测试集数据对**所有训练数据**的距离（用欧几里得距离），将**计算好的距离**与**训练集标签**绑定在一块进行保存。

⑤ 对保存好的 **(距离,训练集标签)** 从小到大排序，取前$K$个（即距离最近的邻居数），统计其**训练集标签** 出现的频数。

⑥ 将频数最高的**训练集标签**保存到**预测标签**结果集中，判断**预测标签**与**原有测试集标签**是否相等，相等即为预测正确，统计数量。

⑦ 计算概率（**预测标签正确的总数量 / 测试集总数**），打印结果。

⑧ 重复**③④⑤⑥⑦**，直到遍历完所有$K$值。

**描述整个过程的伪代码：**

```cpp
label = <A,B,C>; // 定义标签
testSet[55] = (value[4],label); // 特征值,测试集标签
trainSet[95] = (value[4],label); // 特征值,训练集标签
distance[95] = (value,label); // 距离值,绑定的训练集集标签
// 假设数据已经打乱好了 按55:95分割
testSet = readData(1 -> 55); // 读取测试集数据
trainSet = readData(1 -> 95); // 读取训练集数据

forecastLabel[55] = {0}; // 保存预测结果的标签
int count,countA,countB,countC = 0;
for K: 1 -> 15 && (K%2!=0) { // K取奇数
    for testSet,forecastLabel: 1 -> 55 { // 遍历测试集
        for trainSet,distance: 1 -> 95 { // 遍历训练集
            // 计算欧几里得距离
            disVal = EuclideanDistance(testSet.value,trainSet.value,4);
            // 将训练集标签与计算好的距离绑定在一块保存
            distance.add(disVal,trainSet.label<A,B,C>);
        }
        sort(distance.value);
        // 取前K个并统计每个标签出现的频数
        countA,countB,countC = countLabel(distance.label<A,B,C>,K);
        // 将频数最高的训练集标签保存到预测标签结果集中
        forecastLabel.add(max(countA,countB,countC)<A,B,C>);
        // 判断预测标签与原有测试集标签是否相等
        if forecastLabel<A,B,C> == testSet.label<A,B,C> {
            count++; // 统计预测正确的标签数量
        }
    }
    // 打印结果
    print(testSet.label); // 原有测试集标签
    print(forecastLabel); // 预测的标签
    print("K=" + K + " P:" + (count / 55)); // K值 与 正确率
    count,countA,countB,countC = 0; // 重置
    maxLabel = {0}; // 重置
}
```



# 运行结果

(部分)

| ![在这里插入图片描述](https://img-blog.csdnimg.cn/42b39a3fdbfc45dd829b2fb687ad83c0.png) | ![在这里插入图片描述](https://img-blog.csdnimg.cn/a7d0f50eeb5548acaa5d93b4a4b2a6f2.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |



对于每一个$K(1\leq K \leq 15,K\%2\neq0)$值，预测正确的概率：

![打印结果](https://img-blog.csdnimg.cn/d06062b88d3843fda73b3532191813b8.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5aSn54aK5Lq6,size_20,color_FFFFFF,t_70,g_se,x_16)



Email: daxiongren@foxmail.com

WX: zzcxy9

QQ: 1716702942

