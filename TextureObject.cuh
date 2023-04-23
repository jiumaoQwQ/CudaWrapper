#pragma once

#include "helper_cuda.h"

template <class T>
struct TextAcc
{
    cudaTextureObject_t m_textureObject;

    __device__ T read(float x, float y, float z)
    {
        return tex3D<T>(m_textureObject, x, y, z);
    }

    //we use linear Filtering so we should add 0.5
    __device__ T readLinear(float x, float y, float z)
    {
        return tex3D<T>(m_textureObject, x + 0.5, y + 0.5, z + 0.5);
    }
};

template <class T>
struct SurfAcc
{
    cudaSurfaceObject_t m_surfaceObject;

    __device__ T read(int x, int y, int z)
    {
        return surf3Dread<T>(m_textureObject, x, y, z);
    }

    __device__ void write(T val, int x, int y, int z)
    {
        surf3Dwrite(val, m_surfaceObject, x, y, z);
    }
};

template <class T>
class TextureObject
{
public:
    TextureObject(uint3 dim) : m_dim(dim)
    {
        // malloc array
        cudaExtent extent = make_cudaExtent(m_dim.x, m_dim.y, m_dim.z);
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
        checkCudaErrors(cudaMalloc3DArray(&m_array, &channelDesc, extent, cudaArraySurfaceLoadStore));

        // create texture object
        cudaResourceDesc resDesc{};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = m_array;

        cudaTextureDesc textureDesc{};

        for (int i = 0; i < 3; i++)
            textureDesc.addressMode[i] = cudaAddressModeClamp;
        textureDesc.filterMode = cudaFilterModeLinear;
        textureDesc.readMode = cudaReadModeElementType;
        textureDesc.normalizedCoords = false;

        checkCudaErrors(cudaCreateTextureObject(&m_textureObject, &resDesc, &textureDesc, NULL));

        // create surface object
        checkCudaErrors(cudaCreateSurfaceObject(&m_surfaceObject, &resDesc));
    }
    void copyIn(T *raw_ptr)
    {
        // copy resourse
        cudaMemcpy3DParms copy3DParams{};
        copy3DParams.srcPtr = make_cudaPitchedPtr(raw_ptr, m_dim.x * sizeof(T), m_dim.x, m_dim.y);
        copy3DParams.dstArray = m_array;
        copy3DParams.extent = make_cudaExtent(m_dim.x, m_dim.y, m_dim.z);
        copy3DParams.kind = cudaMemcpyHostToDevice;
        checkCudaErrors(cudaMemcpy3D(&copy3DParams));
    }
    void copyOut(T *raw_ptr)
    {
        cudaMemcpy3DParms copy3DParams{};
        copy3DParams.srcArray = m_array;
        copy3DParams.dstPtr = make_cudaPitchedPtr((void *)raw_ptr, m_dim.x * sizeof(T), m_dim.x, m_dim.y);
        copy3DParams.extent = make_cudaExtent(m_dim.x, m_dim.y, m_dim.z);
        copy3DParams.kind = cudaMemcpyDeviceToHost;
        checkCudaErrors(cudaMemcpy3D(&copy3DParams));
    }

    ~TextureObject()
    {
        checkCudaErrors(cudaDestroySurfaceObject(m_surfaceObject));
        checkCudaErrors(cudaDestroyTextureObject(m_textureObject));
        checkCudaErrors(cudaFreeArray(m_array));
    }

    TextAcc<T> getTextAcc()
    {
        return {m_textureObject};
    }

    SurfAcc<T> getSurfAcc()
    {
        return {m_surfaceObject};
    }

private:
    cudaArray_t m_array{};
    uint3 m_dim{};
    cudaTextureObject_t m_textureObject{};
    cudaSurfaceObject_t m_surfaceObject{};
};