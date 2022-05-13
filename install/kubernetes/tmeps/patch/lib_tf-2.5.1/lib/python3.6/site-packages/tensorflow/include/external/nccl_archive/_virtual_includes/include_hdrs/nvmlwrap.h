/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_NVMLWRAP_H_
#define NCCL_NVMLWRAP_H_

#include "nccl.h"

// The NVML library doesn't appear to be thread safe
#include <pthread.h>
extern pthread_mutex_t nvmlLock;
#define NVMLLOCK() pthread_mutex_lock(&nvmlLock)
#define NVMLUNLOCK() pthread_mutex_unlock(&nvmlLock)

#define NVMLLOCKCALL(cmd, ret) do {                      \
    NVMLLOCK();                                          \
    ret = cmd;                                           \
    NVMLUNLOCK();                                        \
} while(false)

#define NVMLCHECK(cmd) do {                              \
    nvmlReturn_t e;                                      \
    NVMLLOCKCALL(cmd, e);                                \
    if( e != NVML_SUCCESS ) {                            \
      WARN("NVML failure '%s'", nvmlErrorString(e));     \
      return ncclSystemError;                            \
    }                                                    \
} while(false)

//#define NVML_DIRECT 1
#ifdef NVML_DIRECT
#include "nvml.h"

static ncclResult_t wrapNvmlSymbols(void) { return ncclSuccess; }
static ncclResult_t wrapNvmlInit(void) { NVMLCHECK(nvmlInit()); return ncclSuccess; }
static ncclResult_t wrapNvmlShutdown(void) { NVMLCHECK(nvmlShutdown()); return ncclSuccess; }
static ncclResult_t wrapNvmlDeviceGetHandleByPciBusId(const char* pciBusId, nvmlDevice_t* device) {
  NVMLCHECK(nvmlDeviceGetHandleByPciBusId(pciBusId, device));
  return ncclSuccess;
}
static ncclResult_t wrapNvmlDeviceGetIndex(nvmlDevice_t device, unsigned* index) {
  NVMLCHECK(nvmlDeviceGetIndex(device, index));
  return ncclSuccess;
}
static ncclResult_t wrapNvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t *isActive) {
  NVMLCHECK(nvmlDeviceGetNvLinkState(device, link, isActive));
  return ncclSuccess;
}
static ncclResult_t wrapNvmlDeviceGetNvLinkRemotePciInfo(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci) {
  NVMLCHECK(nvmlDeviceGetNvLinkRemotePciInfo(device, link, pci));
  return ncclSuccess;
}
static ncclResult_t wrapNvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link,
                                                   nvmlNvLinkCapability_t capability, unsigned int *capResult) {
  NVMLCHECK(nvmlDeviceGetNvLinkCapability(device, link, capability, capResult));
  return ncclSuccess;
}
static ncclResult_t wrapNvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int* major, int* minor) {
  NVMLCHECK(nvmlDeviceGetCudaComputeCapability(device, major, minor));
  return ncclSuccess;
}
#else
// Dynamically handle dependencies on NVML

/* Extracted from nvml.h */
typedef struct nvmlDevice_st* nvmlDevice_t;
#define NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE   16

typedef enum nvmlEnableState_enum
{
    NVML_FEATURE_DISABLED    = 0,     //!< Feature disabled
    NVML_FEATURE_ENABLED     = 1      //!< Feature enabled
} nvmlEnableState_t;

typedef enum nvmlNvLinkCapability_enum
{
    NVML_NVLINK_CAP_P2P_SUPPORTED = 0,     // P2P over NVLink is supported
    NVML_NVLINK_CAP_SYSMEM_ACCESS = 1,     // Access to system memory is supported
    NVML_NVLINK_CAP_P2P_ATOMICS   = 2,     // P2P atomics are supported
    NVML_NVLINK_CAP_SYSMEM_ATOMICS= 3,     // System memory atomics are supported
    NVML_NVLINK_CAP_SLI_BRIDGE    = 4,     // SLI is supported over this link
    NVML_NVLINK_CAP_VALID         = 5,     // Link is supported on this device
    // should be last
    NVML_NVLINK_CAP_COUNT
} nvmlNvLinkCapability_t;

typedef enum nvmlReturn_enum
{
    NVML_SUCCESS = 0,                   //!< The operation was successful
    NVML_ERROR_UNINITIALIZED = 1,       //!< NVML was not first initialized with nvmlInit()
    NVML_ERROR_INVALID_ARGUMENT = 2,    //!< A supplied argument is invalid
    NVML_ERROR_NOT_SUPPORTED = 3,       //!< The requested operation is not available on target device
    NVML_ERROR_NO_PERMISSION = 4,       //!< The current user does not have permission for operation
    NVML_ERROR_ALREADY_INITIALIZED = 5, //!< Deprecated: Multiple initializations are now allowed through ref counting
    NVML_ERROR_NOT_FOUND = 6,           //!< A query to find an object was unsuccessful
    NVML_ERROR_INSUFFICIENT_SIZE = 7,   //!< An input argument is not large enough
    NVML_ERROR_INSUFFICIENT_POWER = 8,  //!< A device's external power cables are not properly attached
    NVML_ERROR_DRIVER_NOT_LOADED = 9,   //!< NVIDIA driver is not loaded
    NVML_ERROR_TIMEOUT = 10,            //!< User provided timeout passed
    NVML_ERROR_IRQ_ISSUE = 11,          //!< NVIDIA Kernel detected an interrupt issue with a GPU
    NVML_ERROR_LIBRARY_NOT_FOUND = 12,  //!< NVML Shared Library couldn't be found or loaded
    NVML_ERROR_FUNCTION_NOT_FOUND = 13, //!< Local version of NVML doesn't implement this function
    NVML_ERROR_CORRUPTED_INFOROM = 14,  //!< infoROM is corrupted
    NVML_ERROR_GPU_IS_LOST = 15,        //!< The GPU has fallen off the bus or has otherwise become inaccessible
    NVML_ERROR_RESET_REQUIRED = 16,     //!< The GPU requires a reset before it can be used again
    NVML_ERROR_OPERATING_SYSTEM = 17,   //!< The GPU control device has been blocked by the operating system/cgroups
    NVML_ERROR_LIB_RM_VERSION_MISMATCH = 18,   //!< RM detects a driver/library version mismatch
    NVML_ERROR_IN_USE = 19,             //!< An operation cannot be performed because the GPU is currently in use
    NVML_ERROR_UNKNOWN = 999            //!< An internal driver error occurred
} nvmlReturn_t;

typedef struct nvmlPciInfo_st
{
    char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE]; //!< The tuple domain:bus:device.function PCI identifier (&amp; NULL terminator)
    unsigned int domain;             //!< The PCI domain on which the device's bus resides, 0 to 0xffff
    unsigned int bus;                //!< The bus on which the device resides, 0 to 0xff
    unsigned int device;             //!< The device's id on the bus, 0 to 31
    unsigned int pciDeviceId;        //!< The combined 16-bit device id and 16-bit vendor id

    // Added in NVML 2.285 API
    unsigned int pciSubSystemId;     //!< The 32-bit Sub System Device ID

    // NVIDIA reserved for internal use only
    unsigned int reserved0;
    unsigned int reserved1;
    unsigned int reserved2;
    unsigned int reserved3;
} nvmlPciInfo_t;
/* End of nvml.h */

ncclResult_t wrapNvmlSymbols(void);

ncclResult_t wrapNvmlInit(void);
ncclResult_t wrapNvmlShutdown(void);
ncclResult_t wrapNvmlDeviceGetHandleByPciBusId(const char* pciBusId, nvmlDevice_t* device);
ncclResult_t wrapNvmlDeviceGetIndex(nvmlDevice_t device, unsigned* index);
ncclResult_t wrapNvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device);
ncclResult_t wrapNvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t *isActive);
ncclResult_t wrapNvmlDeviceGetNvLinkRemotePciInfo(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci);
ncclResult_t wrapNvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link,
                                                   nvmlNvLinkCapability_t capability, unsigned int *capResult);
ncclResult_t wrapNvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int* major, int* minor);

#endif // NVML_DIRECT

#endif // End include guard
