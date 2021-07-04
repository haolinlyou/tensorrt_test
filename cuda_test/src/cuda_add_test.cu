/**
**********************************************************************************************************************************************************************************************************************************
* @file:	cuda_add_test.cu
* @author:	lk
* @email:	lk123400@163.com
* @date:	2021-06-20 13:56:54 Sunday
* @brief:	test cuda add
**********************************************************************************************************************************************************************************************************************************
**/
#include "cuda_runtime.h"
#include <iostream>

using namespace std;


void get_property() 
{
    int dev = 0;
    cudaDeviceProp devProp;
    //CHECK(cudaGetDeviceProperties(&devProp, dev));
    std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
    std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
    std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "每个EM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;

}


// 两个向量加法kernel，grid和block均为一维
__global__ void add(float* x, float * y, float* z, int n)
{
    // 获取全局索引
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // 步长
    int stride = blockDim.x * gridDim.x;
    // printf("%d\t%d\t%d\t%d\t%d\n", threadIdx.x, blockIdx.x, blockDim.x, gridDim.x, stride);
    for (int i = index; i < n; i += stride)
    {
        z[i] = x[i] + y[i];
    }
}


int main()
{

    get_property();
    int N = 1 << 20;
    int nBytes = N * sizeof(float);

    // 申请托管内存
    float *x, *y, *z;
    cudaMallocManaged((void**)&x, nBytes);
    cudaMallocManaged((void**)&y, nBytes);
    cudaMallocManaged((void**)&z, nBytes);

    // 初始化数据
    for (int i = 0; i < N; ++i)
    {
        x[i] = 10.0;
        y[i] = 20.0;
    }

    cout << "threadIdx.x\tblockIdx.x\tblockDim.x\tgridDim.x\tstride\n";
    // 定义kernel的执行配置
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    printf("%d %d \n", gridSize, blockSize);
    // 执行kernel
    add << < gridSize, blockSize >> >(x, y, z, N);

    // 同步device 保证结果能正确访问
    cudaDeviceSynchronize();
    // 检查执行结果
    float maxError = 0.0;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(z[i] - 30.0));
    std::cout << "最大误差: " << maxError << std::endl;

    // 释放内存
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    return 0;
}