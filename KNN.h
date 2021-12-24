/**
 * @file KNN.h
 * @author 大熊人 (daxiongren@foxmail.com)
 * @brief 头文件
 * @version 1.0
 * @date 2021-11-28
 * @copyright Copyright (c) 2021
 * wx:zzcxy9
 * qq:1716702942
 */
#ifndef __KNN_H
#define __KNN_H
#define TOTAL 150     // 总数据的数量
#define TEST_SIZE 55  // 测试数据的数量
#define TRAIN_SIZE 95 // 训练数据的数量
#define N 4           // 特征数据的数量（维数）
#define KN 15         // K的最大取值

/* 距离结构体 */
typedef struct
{
    double value; // 距离数据
    char label;   // 用于绑定训练集标签
} Distance;

/* 鸢尾花结构体 */
typedef struct
{
    double value[N]; // 每种花的4个特征数据
    char type[20];   // 存放花的种类
    char label;      // 用于设置标签 为了方便检测
} Iris;

/* 程序函数接口声明 */
void labelABC(char *type, char *label);
void makeRand(Iris iris[], int n);
void openDataFile(char *path);
void printData();
void loadData();
double EuclideanDistance(double d1[], double d2[], int n);
char compareLabel(int a, int b, int c);
char countLabel(int *count, int k, char forecastLabel);
int cmp(const void *d1, const void *d2);
void printResult(int k, int count);

#endif