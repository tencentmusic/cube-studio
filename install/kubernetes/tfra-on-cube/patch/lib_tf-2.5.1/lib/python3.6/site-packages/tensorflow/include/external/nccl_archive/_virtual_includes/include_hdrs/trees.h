/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TREES_H_
#define NCCL_TREES_H_

ncclResult_t ncclGetBtree(int nranks, int rank, int* u0, int* d1, int* d0, int* parentChildType);
ncclResult_t ncclGetDtree(int nranks, int rank, int* u0, int* d0_0, int* d0_1, int* parentChildType0, int* u1, int* d1_0, int* d1_1, int* parentChildType1);

#endif
