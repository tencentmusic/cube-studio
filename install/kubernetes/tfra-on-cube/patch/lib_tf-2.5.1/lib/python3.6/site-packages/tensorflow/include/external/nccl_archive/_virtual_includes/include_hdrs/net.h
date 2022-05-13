/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INT_NET_H_
#define NCCL_INT_NET_H_

#include "nccl.h"
#include "nccl_net.h"

extern ncclNet_t* ncclNet;
typedef char ncclNetHandle_t[NCCL_NET_HANDLE_MAXSIZE];

// Translation to external API
static const char* ncclNetName() { return ncclNet->name; }
static ncclResult_t ncclNetDevices(int* ndev) { NCCLCHECK(ncclNet->devices(ndev)); return ncclSuccess; }
static ncclResult_t ncclNetGetProperties(int dev, ncclNetProperties_t* props) { NCCLCHECK(ncclNet->getProperties(dev, props)); return ncclSuccess; }
static ncclResult_t ncclNetListen(int dev, void* handle, void** listenComm) { NCCLCHECK(ncclNet->listen(dev, handle, listenComm)); return ncclSuccess; }
static ncclResult_t ncclNetConnect(int dev, void* handle, void** sendComm) { NCCLCHECK(ncclNet->connect(dev, handle, sendComm)); return ncclSuccess; }
static ncclResult_t ncclNetAccept(void* listenComm, void** recvComm) { NCCLCHECK(ncclNet->accept(listenComm, recvComm)); return ncclSuccess; }
static ncclResult_t ncclNetRegMr(void* comm, void* data, int size, int type, void** mhandle) { NCCLCHECK(ncclNet->regMr(comm, data, size, type, mhandle)); return ncclSuccess; }
static ncclResult_t ncclNetDeregMr(void* comm, void* mhandle) { NCCLCHECK(ncclNet->deregMr(comm, mhandle)); return ncclSuccess; }
static ncclResult_t ncclNetIsend(void* sendComm, void* data, int size, void* mhandle, void** request) { NCCLCHECK(ncclNet->isend(sendComm, data, size, mhandle, request)); return ncclSuccess; }
static ncclResult_t ncclNetIrecv(void* recvComm, void* data, int size, void* mhandle, void** request) { NCCLCHECK(ncclNet->irecv(recvComm, data, size, mhandle, request)); return ncclSuccess; }
static ncclResult_t ncclNetIflush(void* recvComm, void* data, int size, void* mhandle, void** request) { NCCLCHECK(ncclNet->iflush(recvComm, data, size, mhandle, request)); return ncclSuccess; }
static ncclResult_t ncclNetTest(void* request, int* done, int* size) { NCCLCHECK(ncclNet->test(request, done, size)); return ncclSuccess; }
static ncclResult_t ncclNetCloseSend(void* sendComm) { NCCLCHECK(ncclNet->closeSend(sendComm)); return ncclSuccess; }
static ncclResult_t ncclNetCloseRecv(void* recvComm) { NCCLCHECK(ncclNet->closeRecv(recvComm)); return ncclSuccess; }
static ncclResult_t ncclNetCloseListen(void* listenComm) { NCCLCHECK(ncclNet->closeListen(listenComm)); return ncclSuccess; }

// Test whether the current GPU support GPU Direct RDMA.
#define GPU_BUF_SIZE (2*1024*1024)
static ncclResult_t ncclGpuGdrSupport(int* gdrSupport) {
  int netDevs;
  NCCLCHECK(ncclNetDevices(&netDevs));
  *gdrSupport = 0;
  for (int dev=0; dev<netDevs; dev++) {
    // Find a net device which is GDR-capable
    ncclNetProperties_t props;
    NCCLCHECK(ncclNet->getProperties(dev, &props));
    if ((props.ptrSupport & NCCL_PTR_CUDA) == 0) continue;

    // Allocate memory on the GPU and try to register it on the NIC.
    void *lComm = NULL, *sComm = NULL, *rComm = NULL;
    ncclNetHandle_t handle;
    void* gpuPtr = NULL;
    void* mHandle = NULL;
    NCCLCHECK(ncclNetListen(dev, &handle, &lComm));
    NCCLCHECK(ncclNetConnect(dev, &handle, &sComm));
    NCCLCHECK(ncclNetAccept(lComm, &rComm));
    CUDACHECK(cudaMalloc(&gpuPtr, GPU_BUF_SIZE));
    ncclDebugNoWarn = NCCL_NET;
    if (ncclNetRegMr(sComm, gpuPtr, GPU_BUF_SIZE, NCCL_PTR_CUDA, &mHandle) == ncclSuccess) {
      NCCLCHECK(ncclNetDeregMr(sComm, mHandle));
      NCCLCHECK(ncclNetRegMr(rComm, gpuPtr, GPU_BUF_SIZE, NCCL_PTR_CUDA, &mHandle));
      NCCLCHECK(ncclNetDeregMr(rComm, mHandle));
      *gdrSupport = 1;
    }
    ncclDebugNoWarn = 0;
    CUDACHECK(cudaFree(gpuPtr));
    NCCLCHECK(ncclNetCloseRecv(rComm));
    NCCLCHECK(ncclNetCloseSend(sComm));
    NCCLCHECK(ncclNetCloseListen(lComm));
    break;
  }
  return ncclSuccess;
}

extern ncclNet_t ncclNetIb;
extern ncclNet_t ncclNetSocket;

#endif
