import React, { useState } from 'react';
import { Icon, Layer } from '@fluentui/react';
import { Handle, Position, NodeProps } from 'react-flow-renderer';
import Model from '../Model';
import style from './style';
import { message, Modal, Radio, Space } from 'antd';
import storage from '@src/utils/storage';
import { useAppDispatch, useAppSelector } from '@src/models/hooks';
import { savePipeline, selectPipelineId, updateEditing } from '@src/models/pipeline';
import { updateLoading } from '@src/models/task';
import api from '@src/api';
import { selectElements, updateElements } from '@src/models/element';
import { updateErrMsg } from '@src/models/app';
import { useTranslation } from 'react-i18next';

const DataSet: React.FC<NodeProps> = props => {
  const [visible, setVisible] = useState(false)
  const [recommendList, setRecommendList] = useState<any[]>([])
  const [currentRecommend, setCurrentRecommend] = useState<any>()
  const jobTemplate = storage.get('job_template');
  const dataMap = (jobTemplate.value || []).reduce((pre: any, next: any) => ({ ...pre, [next.name]: next }), {})
  const pipelineId = useAppSelector(selectPipelineId);
  const elements = useAppSelector(selectElements);
  const dispatch = useAppDispatch();
  const { t, i18n } = useTranslation();

  return (
    <>
      <Modal title={t('智能推荐下游节点')} visible={visible} onCancel={() => {
        setCurrentRecommend(undefined)
        setVisible(false)
      }} onOk={() => {
        if (!currentRecommend) {
          message.warn(t('请先选择推荐节点'))
          return
        }

        const modelInfo = dataMap[currentRecommend]
        const args = JSON.parse(modelInfo.args);
        const defaultArgs = {};

        Object.keys(args).reduce((acc, cur) => {
          const curArgs = {};

          Object.keys(args[cur]).reduce((curAcc, argsKey) => {
            const defaultValue = args[cur][argsKey].default;
            Object.assign(curAcc, { [argsKey]: defaultValue });
            return curAcc;
          }, curArgs);
          Object.assign(acc, curArgs);

          return acc;
        }, defaultArgs);

        const position = {
          x: props.xPos || 0,
          y: (+(props?.yPos || 0)) + 100
        }

        if (pipelineId) {
          dispatch(updateLoading(true));
          const taskName = `${modelInfo.name.replace(/\.|[\u4e00-\u9fa5]/g, '').replace(/_|\s/g, '-') || 'task'
            }-${Date.now()}`.substring(0, 49);
          api
            .task_modelview_add(pipelineId, {
              job_template: modelInfo.id,
              pipeline: +pipelineId,
              name: taskName,
              label: `${t('新建')} ${modelInfo.name} ${t('任务')}`,
              volume_mount: 'kubeflow-user-workspace(pvc):/mnt',
              image_pull_policy: 'Always',
              working_dir: '',
              command: '',
              overwrite_entrypoint: 0,
              node_selector: 'cpu=true,train=true',
              resource_memory: '2G',
              resource_cpu: '2',
              resource_gpu: '0',
              resource_rdma: '0',
              timeout: 0,
              retry: 0,
              args: JSON.stringify(defaultArgs),
            })
            .then((res: any) => {
              if (res?.result?.id) {
                const newNode = {
                  id: `${res.result.id}`,
                  type: 'dataSet',
                  position,
                  data: {
                    info: modelInfo,
                    name: taskName,
                    label: `${t('新建')} ${modelInfo.name} ${t('任务')}`,
                  },
                };
                dispatch(updateEditing(true));
                dispatch(updateElements(elements.concat(newNode)));
                setTimeout(() => {
                  dispatch(savePipeline());
                }, 2000);
              }
            })
            .catch(err => {
              if (err.response) {
                dispatch(updateErrMsg({ msg: err.response.data.message }));
              }
            })
            .finally(() => {
              dispatch(updateLoading(false));
              setVisible(false)
            });
        }
      }}>
        <div>
          <Radio.Group onChange={(e) => { setCurrentRecommend(e.target.value) }} value={currentRecommend}>
            <Space direction="vertical">
              {
                recommendList.map((item: any) => {
                  return <Radio value={item.name} key={`recommend_${item.name}`}>{item.name}</Radio>
                })
              }
            </Space>
          </Radio.Group>
        </div>
      </Modal>
      <Handle type="target" position={Position.Top} className={style.handleStyle} />
      <div className={props.selected ? style.nodeOnSelect : style.nodeContainer} style={{ borderColor: props.selected ? props?.data?.info?.color?.color : '', backgroundColor: props.selected ? props?.data?.info?.color?.bg : '' }}>
        <div className={style.nodeIconWrapper} style={{ backgroundColor: props?.data?.info?.color?.color }}>
          <Icon iconName="Database" className={style.nodeIcon}></Icon>
        </div>
        <div className={style.nodeContentWrapper}>
          <div className={style.nodeTitle}>{props?.data?.label}</div>
          {
            props?.data?.info ? <div className={style.nodeTips}>{props?.data?.info['describe']}</div> : null
          }
        </div>
      </div>
      <Handle type="source" onClick={() => {
        console.log('props', props)
        let recommendObj = props.data?.info?.expand;
        if (Object.prototype.toString.call(recommendObj) === '[object String]') {
          recommendObj = JSON.parse(props.data?.info?.expand || '{}')
        }
        const recommend = recommendObj.rec_job_template
        if (recommend) {
          setRecommendList([dataMap[recommend]])
          setVisible(true)
        }
      }} position={Position.Bottom} className={style.handleStyle} />
      {/* 配置板 */}
      <Layer hostId="hostId_model" className="data-set-layer">
        <Model model={props}></Model>
      </Layer>
    </>
  );
};

export default DataSet;
