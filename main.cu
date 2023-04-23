#include "TextureObject.cuh"
#include <vector>

#include <iostream>

template <class T>
__global__ void process(TextAcc<T> acc)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    float num = acc.readLinear(x, y, z);
    if (x == 1 && y == 1 && z == 1)
        printf("%f ", num);
}

int main()
{
    std::vector<float> vec(128 * 128 * 128);
    for (int i = 0; i < 128 * 128 * 128; i++)
    {
        vec[i] = i;
    }
    TextureObject<float> obj({128, 128, 128});
    obj.copyIn(vec.data());
    process<<<{128 / 8, 128 / 8, 128 / 8}, {8, 8, 8}>>>(obj.getTextAcc());

    return 0;
}