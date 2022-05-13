/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_SHM_H_
#define NCCL_SHM_H_

#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

// Change functions behavior to match other SYS functions
static int shm_allocate(int fd, const int shmsize) {
  int err = posix_fallocate(fd, 0, shmsize);
  if (err) { errno = err; return -1; }
  return 0;
}
static int shm_map(int fd, const int shmsize, void** ptr) {
  *ptr = mmap(NULL, shmsize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  return (*ptr == MAP_FAILED) ? -1 : 0;
}

static ncclResult_t shmSetup(const char* shmname, const int shmsize, int* fd, void** ptr, int create) {
  SYSCHECKVAL(shm_open(shmname, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR), "shm_open", *fd);
  if (create) SYSCHECK(shm_allocate(*fd, shmsize), "posix_fallocate");
  SYSCHECK(shm_map(*fd, shmsize, ptr), "mmap");
  close(*fd);
  *fd = -1;
  if (create) memset(*ptr, 0, shmsize);
  return ncclSuccess;
}

static ncclResult_t shmOpen(const char* shmname, const int shmsize, void** shmPtr, void** devShmPtr, int create) {
  int fd = -1;
  void* ptr = MAP_FAILED;
  ncclResult_t res = ncclSuccess;

  NCCLCHECKGOTO(shmSetup(shmname, shmsize, &fd, &ptr, create), res, sysError);
  CUDACHECKGOTO(cudaHostRegister(ptr, shmsize, cudaHostRegisterMapped), res, cudaError);
  CUDACHECKGOTO(cudaHostGetDevicePointer(devShmPtr, ptr, 0), res, cudaError);

  *shmPtr = ptr;
  return ncclSuccess;
sysError:
  WARN("Error while %s shared memory segment %s (size %d)\n", create ? "creating" : "attaching to", shmname, shmsize);
cudaError:
  if (fd != -1) close(fd);
  if (create) shm_unlink(shmname);
  if (ptr != MAP_FAILED) munmap(ptr, shmsize);
  *shmPtr = NULL;
  return res;
}

static ncclResult_t shmUnlink(const char* shmname) {
  if (shmname != NULL) SYSCHECK(shm_unlink(shmname), "shm_unlink");
  return ncclSuccess;
}

static ncclResult_t shmClose(void* shmPtr, void* devShmPtr, const int shmsize) {
  CUDACHECK(cudaHostUnregister(shmPtr));
  if (munmap(shmPtr, shmsize) != 0) {
    WARN("munmap of shared memory failed");
    return ncclSystemError;
  }
  return ncclSuccess;
}

#endif
