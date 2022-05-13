/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_ENQUEUE_H_
#define NCCL_ENQUEUE_H_

#include "comm.h"
#include "group.h"
#include "collectives.h"

ncclResult_t ncclEnqueueCheck(struct ncclInfo* info);
ncclResult_t ncclCpuBarrierIn(struct ncclComm* comm, int* isLast);
ncclResult_t ncclCpuBarrierLast(struct ncclComm* comm);
ncclResult_t ncclCpuBarrierOut(struct ncclComm* comm);
ncclResult_t ncclBarrierEnqueue(struct ncclComm* comm);
ncclResult_t ncclBarrierEnqueueWait(struct ncclComm* comm);
ncclResult_t ncclEnqueueEvents(struct ncclComm* comm);
ncclResult_t ncclSaveKernel(struct ncclInfo* info);
ncclResult_t ncclSaveP2pKernel(struct ncclInfo* info);
ncclResult_t ncclSaveCommKernels(struct ncclComm* comm);

#endif // End include guard
