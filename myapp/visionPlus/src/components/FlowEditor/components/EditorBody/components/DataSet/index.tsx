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
              volume_mount: 'kubeflow-user-workspace(pvc):/mnt,kubeflow-archives(pvc):/archives',
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

      <Handle type="target" position={Position.Top} style={{ top: '-7px', '--hover-color': props?.data?.info?.color?.color } as any} className={style.handleStyle} />
      <div className={props.selected ? style.nodeOnSelect : style.nodeContainer} style={{ borderColor: props.selected ? props?.data?.info?.color?.color : '', backgroundColor: props.selected ? props?.data?.info?.color?.bg : '' }}>
        <div className={style.nodeIconWrapper} style={{ backgroundColor: props?.data?.info?.color?.color }}>
          {/* <div dangerouslySetInnerHTML={{ __html: '<svg t="1645412532834" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="2039" width="200" height="200"><path d="M448.544 496H256a32 32 0 0 0 0 64l146.976-0.192-233.6 233.568a32 32 0 0 0 45.248 45.248l233.664-233.632v147.264a32 32 0 1 0 64 0v-192.512a63.84 63.84 0 0 0-63.744-63.744M838.624 201.376a31.968 31.968 0 0 0-45.248 0L576 418.752V272a32 32 0 0 0-64 0v192.544c0 35.136 28.608 63.712 63.744 63.712h192.512a32 32 0 1 0 0-64l-147.488 0.224 217.856-217.856a31.968 31.968 0 0 0 0-45.248" p-id="2040"></path></svg>' }}></div> */}
          <Icon iconName="Database" className={style.nodeIcon}></Icon>
        </div>
        <div className={style.nodeContentWrapper}>
          <div className={style.nodeTitle}>{props?.data?.label}</div>
          <div className={style.nodeTips}>{props?.data?.info['template-group']} - {props?.data?.info['template']}</div>
        </div>
      </div>
      <Handle onClick={() => {
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
      }} type="source" position={Position.Bottom} className={style.handleStyle} style={{ '--hover-color': props?.data?.info?.color?.color } as any} />
      {/* 配置板 */}
      <Layer hostId="hostId_model" className="data-set-layer">
        <Model model={props}></Model>
      </Layer>
    </>
  );
};

export default DataSet;
