/*************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PARAM_H_
#define NCCL_PARAM_H_

#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>

static const char* userHomeDir() {
  struct passwd *pwUser = getpwuid(getuid());
  return pwUser == NULL ? NULL : pwUser->pw_dir;
}

static void setEnvFile(const char* fileName) {
  FILE * file = fopen(fileName, "r");
  if (file == NULL) return;

  char *line = NULL;
  char envVar[1024];
  char envValue[1024];
  size_t n = 0;
  ssize_t read;
  while ((read = getline(&line, &n, file)) != -1) {
    if (line[read-1] == '\n') line[read-1] = '\0';
    int s=0; // Env Var Size
    while (line[s] != '\0' && line[s] != '=') s++;
    if (line[s] == '\0') continue;
    strncpy(envVar, line, std::min(1023,s));
    envVar[s] = '\0';
    s++;
    strncpy(envValue, line+s, 1023);
    envValue[1023]='\0';
    setenv(envVar, envValue, 0);
  }
  if (line) free(line);
  fclose(file);
}

static void initEnv() {
  char confFilePath[1024];
  const char * userDir = userHomeDir();
  if (userDir) {
    sprintf(confFilePath, "%s/.nccl.conf", userDir);
    setEnvFile(confFilePath);
  }
  sprintf(confFilePath, "/etc/nccl.conf");
  setEnvFile(confFilePath);
}


#define NCCL_PARAM(name, env, default_value) \
pthread_mutex_t ncclParamMutex##name = PTHREAD_MUTEX_INITIALIZER; \
int64_t ncclParam##name() { \
  static_assert(default_value != -1LL, "default value cannot be -1"); \
  static int64_t value = -1LL; \
  pthread_mutex_lock(&ncclParamMutex##name); \
  if (value == -1LL) { \
    value = default_value; \
    char* str = getenv("NCCL_" env); \
    if (str && strlen(str) > 0) { \
      errno = 0; \
      int64_t v = strtoll(str, NULL, 0); \
      if (errno) { \
        INFO(NCCL_ALL,"Invalid value %s for %s, using default %lu.", str, "NCCL_" env, value); \
      } else { \
        value = v; \
        INFO(NCCL_ALL,"%s set by environment to %lu.", "NCCL_" env, value);  \
      } \
    } \
  } \
  pthread_mutex_unlock(&ncclParamMutex##name); \
  return value; \
}

#endif
