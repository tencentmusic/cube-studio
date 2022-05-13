/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdlib.h>

#ifndef NCCL_P2P_H_
#define NCCL_P2P_H_

struct ncclP2Pinfo {
  void* buff;
  ssize_t nbytes;
  struct ncclP2Pinfo* next;
};

struct ncclP2Plist {
  struct ncclP2Pinfo *head;
  struct ncclP2Pinfo *tail;
};

static ncclResult_t enqueueP2pInfo(ncclP2Plist* p2p, void* buff, ssize_t nBytes) {
  if (p2p == NULL) return ncclInternalError;
  struct ncclP2Pinfo* next;
  NCCLCHECK(ncclCalloc(&next, 1));
  next->buff = buff;
  next->nbytes = nBytes;
  if (p2p->tail != NULL) p2p->tail->next = next;
  p2p->tail = next;
  if (p2p->head == NULL) p2p->head = next;
  return ncclSuccess;
}

static ncclResult_t dequeueP2pInfo(ncclP2Plist* p2p) {
  if (p2p == NULL) return ncclInternalError;
  struct ncclP2Pinfo* temp = p2p->head;
  p2p->head = p2p->head->next;
  if (p2p->tail == temp) p2p->tail = NULL;
  free(temp);
  return ncclSuccess;
}
#endif
