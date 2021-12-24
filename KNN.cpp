/**
 * @file KNN.cpp
 * @author 大熊人 (daxiongren@foxmail.com)
 * @brief 用KNN算法简单实现对鸢尾花分类
 * @version 1.0
 * @date 2021-11-28
 * @copyright Copyright (c) 2021
 * wx:zzcxy9
 * qq:1716702942
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "KNN.h"

Iris testSet[TEST_SIZE];	   // 测试集结构体数组
Iris forecastSet[TEST_SIZE];   // 保存预测的标签
Iris trainSet[TRAIN_SIZE];	   // 训练集结构体数组
Iris temp[TOTAL];			   // 临时存放数据结构体数组
Distance distance[TRAIN_SIZE]; // 存放距离结构体数组

/**
 * @brief 把不同种类的花分别转化成 A B C 标签
 * @param type[IN] 花的种类
 * @param label[OUT] 转化的标签
 */
void labelABC(char *type, char *label)
{
	if (strcmp(type, "\"setosa\"") == 0)
		*label = 'A';
	if (strcmp(type, "\"versicolor\"") == 0)
		*label = 'B';
	if (strcmp(type, "\"virginica\"") == 0)
		*label = 'C';
}

/**
 * @brief 利用伪随机数进行数据打乱
 * @param iris
 * @param n
 */
void makeRand(Iris iris[], int n)
{
	Iris t;
	int i, n1, n2;
	srand((unsigned int)time(NULL)); //获取随机数的种子,百度查下用法
	for (i = 0; i < n; i++)
	{
		n1 = (rand() % n); //产生n以内的随机数  n是数组元素个数
		n2 = (rand() % n);
		/* 若两随机数不相等 则下标为这两随机数的数组进行交换 */
		if (n1 != n2)
		{
			t = iris[n1];
			iris[n1] = iris[n2];
			iris[n2] = t;
		}
	}
}

/**
 * @brief 打开数据文件
 * @param path 数据文件的路径
 */
void openDataFile(char *path)
{
	int i, j;
	// 用于先存放150个数据后再打乱
	FILE *fp = NULL;
	fp = fopen(path, "r");
	for (i = 0; i < TOTAL; i++)
	{
		for (j = 0; j < N; j++)
		{
			fscanf(fp, "%lf ", &temp[i].value[j]);
		}
		fscanf(fp, "%s", temp[i].type);
		/* 把不同种类的花分别转化成 A B C 标签 */
		labelABC(temp[i].type, &temp[i].label);
	}
	makeRand(temp, TOTAL); //打乱所有数据
	fclose(fp);
	fp = NULL;
}

/**
 * @brief 把分割后的数据都打印出来  便于观察是否已经打乱
 */
void printData()
{
	int i, j;
	printf("\n设置标签 -> 打乱 -> 按%d/%d分割\n", TEST_SIZE, TRAIN_SIZE);
	printf("数据如下:\n\n");
	printf("%d组测试集:\n", TEST_SIZE);
	for (i = 0; i < TEST_SIZE; i++)
	{
		for (j = 0; j < N; j++)
		{
			printf("%.2lf ", testSet[i].value[j]);
		}
		printf("%c\n", testSet[i].label);
	}
	printf("\n\n%d组训练集:\n", TRAIN_SIZE);
	for (i = 0; i < TRAIN_SIZE; i++)
	{
		for (j = 0; j < N; j++)
		{
			printf("%.2lf ", trainSet[i].value[j]);
		}
		printf("%c\n", trainSet[i].label);
	}
}

/**
 * @brief 加载数据  分割：测试TEST_SIZE组   训练TRAIN_SIZE组
 */
void loadData()
{
	int i, j, n = 0, m = 0;
	for (i = 0; i < TOTAL; i++)
	{
		/* 先将TEST_SIZE个数据存入测试集 */
		if (i < TEST_SIZE)
		{
			for (j = 0; j < N; j++)
			{
				testSet[n].value[j] = temp[i].value[j]; //存入花的四个特征数据
			}
			testSet[n].label = temp[i].label; //存入花的标签
			n++;
		}
		else /* 剩下的数据存入训练集 */
		{
			for (j = 0; j < N; j++)
			{
				trainSet[m].value[j] = temp[i].value[j]; //存入花的四个特征数据
			}
			trainSet[m].label = temp[i].label; //存入花的标签
			m++;
		}
	}
}

/**
 * @brief 计算欧几里得距离
 * @param d1
 * @param d2
 * @param n 维数
 * @return double
 */
double EuclideanDistance(double d1[], double d2[], int n)
{
	double result = 0.0;
	int i;
	/* 欧几里得距离 */
	for (i = 0; i < n; i++)
	{
		result += pow(d1[i] - d2[i], 2.0);
	}
	result = sqrt(result);

	return result; //返回距离
}

/**
 * @brief 比较三个标签出现的频数
 * @param a
 * @param b
 * @param c
 * @return char 返回出现的频数最多的标签
 */
char compareLabel(int a, int b, int c)
{
	if (a > b && a > c)
	{
		return 'A';
	}
	if (b > a && b > c)
	{
		return 'B';
	}
	if (c > a && c > b)
	{
		return 'C';
	}
	return 0;
}

/**
 * @brief 统计与测试集距离最邻近的k个标签出现的频数
 * @param count[OUT] 用于统计
 * @param k[IN] 当前K值
 * @param forecastLabel[IN] 训练集的预测标签
 * @return 返回频数最高的标签
 */
char countLabel(int *count, int k, char forecastLabel)
{
	int i;
	int sumA = 0, sumB = 0, sumC = 0; //分别统计距离最邻近的三类标签出现的频数
	for (i = 0; i < k; i++)
	{
		switch (distance[i].label)
		{
		case 'A':
			sumA++;
			break;
		case 'B':
			sumB++;
			break;
		case 'C':
			sumC++;
			break;
		}
	}
	/* 检测出现频数最高的标签与测试集的预测标签是否相等 */
	char maxLabel = compareLabel(sumA, sumB, sumC);
	if (maxLabel == forecastLabel)
	{
		(*count)++; //统计符合的数量
	}
	return maxLabel;
}

/* 快速排序qsort函数的cmp回调函数 */
int cmp(const void *d1, const void *d2)
{
	Distance D1 = *(Distance *)d1;
	Distance D2 = *(Distance *)d2;
	return D1.value > D2.value ? 1 : -1;
}

/**
 * @brief 打印结果
 * @param k K值
 * @param count 预测正确的总数量
 */
void printResult(int k, int count)
{
	int i;
	printf("对比结果:\n");
	/* 打印每个K值对应的概率 */
	printf("K = %d     P = %.2lf%%\n", k, (100.0 * count) / TEST_SIZE);
	printf("原有标签:");
	printf("[%c", testSet[0].label);
	for (i = 1; i < TEST_SIZE; i++)
		printf(",%c", testSet[i].label);
	printf("]\n");
	printf("预测标签:");
	printf("[%c", forecastSet[0].label);
	for (i = 1; i < TEST_SIZE; i++)
		printf(",%c", forecastSet[i].label);
	printf("]\n\n");
}

int main()
{
	int i, j;
	int k;		   // k值
	int count = 0; //用于统计预测正确的标签数量
	/* openDataFile("你的数据文件路径") 如果放在代码文件路径下那就直接写文件名(建议写绝对路径) */
	openDataFile("D:/OneDrive/Desktop/Study/C++/KNNAlgorithm/iris.txt"); // 打开数据文件 -> 打乱数据
	loadData();															 // 加载打乱后的数据并分割
	printData();														 // 打印数据
	printf("\n\n测试集:%d组  训练集:%d组\n\n", TEST_SIZE, TRAIN_SIZE);

	for (k = 1; k <= KN; k += 2) // k值：1--KN(取奇数)  KN = 15(宏定义)
	{
		for (i = 0; i < TEST_SIZE; i++) // 遍历测试集
		{
			for (j = 0; j < TRAIN_SIZE; j++) // 遍历训练集
			{
				/* 把计算欧几里得距离依次存入distance结构体数组的value中 */
				distance[j].value = EuclideanDistance(testSet[i].value, trainSet[j].value, N);
				/* 将训练集标签与计算好的距离绑定在一块 */
				distance[j].label = trainSet[j].label;
			}
			/* 用qsort函数从小到大排序(距离,训练集标签) */
			qsort(distance, TRAIN_SIZE, sizeof(distance[0]), cmp);
			/* 统计与测试集标签距离最邻近的k个标签出现的频数 并返回频数最后高标签 即预测的标签 */
			forecastSet[i].label = countLabel(&count, k, testSet[i].label);
		}
		/* 打印结果 */
		printResult(k, count);
		count = 0; // 重置
	}
	getchar();
	return 0;
}