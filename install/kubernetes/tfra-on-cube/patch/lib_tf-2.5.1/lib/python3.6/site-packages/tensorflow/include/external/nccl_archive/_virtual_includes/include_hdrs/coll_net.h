/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef COLL_NET_H_
#define COLL_NET_H_

#include "nccl.h"
#include "nccl_net.h"

extern ncclCollNet_t* ncclCollNet;
typedef char collNetHandle_t[NCCL_NET_HANDLE_MAXSIZE];

// Translation to external API
static const char* collNetName() { return ncclCollNet->name; }
static ncclResult_t collNetDevices(int* ndev) { NCCLCHECK(ncclCollNet->devices(ndev)); return ncclSuccess; }
static ncclResult_t collNetGetProperties(int dev, ncclNetProperties_t* props) { NCCLCHECK(ncclCollNet->getProperties(dev, props)); return ncclSuccess; }
static ncclResult_t collNetListen(int dev, void* handle, void** listenComm) { NCCLCHECK(ncclCollNet->listen(dev, handle, listenComm)); return ncclSuccess; }
static ncclResult_t collNetConnect(void* handles[], int nranks, int rank, void* listenComm, void** collComm) { NCCLCHECK(ncclCollNet->connect(handles, nranks, rank, listenComm, collComm)); return ncclSuccess; }
static ncclResult_t collNetReduceSupport(ncclDataType_t dataType, ncclRedOp_t redOp, int* supported) { NCCLCHECK(ncclCollNet->reduceSupport(dataType, redOp, supported)); return ncclSuccess; }
static ncclResult_t collNetRegMr(void* comm, void* data, int size, int type, void** mhandle) { NCCLCHECK(ncclCollNet->regMr(comm, data, size, type, mhandle)); return ncclSuccess; }
static ncclResult_t collNetDeregMr(void* comm, void* mhandle) { NCCLCHECK(ncclCollNet->deregMr(comm, mhandle)); return ncclSuccess; }
static ncclResult_t collNetIallreduce(void* collComm, void* sendData, void* recvData, int count, ncclDataType_t dataType, ncclRedOp_t redOp, void* sendMhandle, void* recvMhandle,  void** request) {
  NCCLCHECK(ncclCollNet->iallreduce(collComm, sendData, recvData, count, dataType, redOp, sendMhandle, recvMhandle, request)); return ncclSuccess; }
static ncclResult_t collNetIflush(void* collComm, void* data, int size, void* mhandle, void** request) { NCCLCHECK(ncclCollNet->iflush(collComm, data, size, mhandle, request)); return ncclSuccess; }
static ncclResult_t collNetTest(void* request, int* done, int* size) { NCCLCHECK(ncclCollNet->test(request, done, size)); return ncclSuccess; }
static ncclResult_t collNetCloseColl(void* collComm) { NCCLCHECK(ncclCollNet->closeColl(collComm)); return ncclSuccess; }
static ncclResult_t collNetCloseListen(void* listenComm) { NCCLCHECK(ncclCollNet->closeListen(listenComm)); return ncclSuccess; }

static int collNetSupport() { return ncclCollNet != NULL ? 1 : 0; }

#endif
