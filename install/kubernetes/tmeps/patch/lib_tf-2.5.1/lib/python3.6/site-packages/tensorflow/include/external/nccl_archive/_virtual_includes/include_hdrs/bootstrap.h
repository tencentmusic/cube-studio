/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_BOOTSTRAP_H_
#define NCCL_BOOTSTRAP_H_

#include "nccl.h"

ncclResult_t bootstrapNetInit();
ncclResult_t bootstrapCreateRoot(ncclUniqueId* commId, bool idFromEnv);
ncclResult_t bootstrapGetUniqueId(ncclUniqueId* out);
ncclResult_t bootstrapInit(ncclUniqueId* id, int rank, int nranks, void** commState);
ncclResult_t bootstrapAllGather(void* commState, void* allData, int size);
ncclResult_t bootstrapSend(void* commState, int peer, void* data, int size);
ncclResult_t bootstrapRecv(void* commState, int peer, void* data, int size);
ncclResult_t bootstrapRemAlloc(size_t size, int rank, void* commState, int* id, cudaIpcMemHandle_t* ipc, void** ptr);
ncclResult_t bootstrapRemFree(int id, int rank, void* commState);
ncclResult_t bootstrapClose(void* commState);
ncclResult_t bootstrapAbort(void* commState);
#endif
