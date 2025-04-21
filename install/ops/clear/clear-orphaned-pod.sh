#!/bin/bash

# 这个脚本使用优雅方式进行pod相关挂载卷的卸载处理和卷目录删除，并不手动删除pod目录（其内可能包含pod数据）

KUBELET_HOME=/var/lib
# 通过系统日志获取到全部孤儿pod的podid
#docker logs --tail 1000 kubelet > kubelet.log    # 这个保存不了日志
# 获取kubelet日志地址
logpath=`docker inspect --format='{{.LogPath}}' kubelet`
tail -n 500 $logpath > kubelet.log
for podid in $(grep "orphaned pod.*podUID" kubelet.log | tail -100 | awk '{print $11}' | sed 's/podUID=//g');
do
    sleep 3
    echo "check $podid"
    if [ ! -d ${KUBELET_HOME}/kubelet/pods/$podid ]; then
        break
    fi

    if [ -d ${KUBELET_HOME}/kubelet/pods/$podid/volume-subpaths/ ]; then
        mountpath=$(mount | grep ${KUBELET_HOME}/kubelet/pods/$podid/volume-subpaths/ | awk '{print $3}')
        for mntPath in $mountpath;
        do
            umount $mntPath
        done
        rm -rf ${KUBELET_HOME}/kubelet/pods/$podid/volume-subpaths
    fi

    csiMounts=$(mount | grep "${KUBELET_HOME}/kubelet/pods/$podid/volumes/kubernetes.io~csi")
    if [ "$csiMounts" != "" ]; then
        echo "csi is mounted at: $csiMounts"
        exit 1
    else
        rm -rf ${KUBELET_HOME}/kubelet/pods/$podid/volumes/kubernetes.io~csi
    fi

    if [ -d ${KUBELET_HOME}/kubelet/pods/$podid/volumes/ ]; then
      volumeTypes=$(ls ${KUBELET_HOME}/kubelet/pods/$podid/volumes/)
      for volumeType in $volumeTypes;
      do
          subVolumes=$(ls -A ${KUBELET_HOME}/kubelet/pods/$podid/volumes/$volumeType)
          if [ "$subVolumes" != "" ]; then
              echo "${KUBELET_HOME}/kubelet/pods/$podid/volumes/$volumeType contents volume: $subVolumes"
              exit 1
          else
              rmdir ${KUBELET_HOME}/kubelet/pods/$podid/volumes/$volumeType
          fi
      done
    fi
done

