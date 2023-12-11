import React, { useState, useEffect, MouseEvent, useRef } from 'react';
import { Stack, LayerHost, Spinner, SpinnerSize } from '@fluentui/react';
import ReactFlow, {
  removeElements,
  addEdge,
  MiniMap,
  Controls,
  Node,
  FlowElement,
  OnLoadParams,
  Elements,
  Connection,
  Edge,
  isEdge,
  isNode,
} from 'react-flow-renderer';
import api from '@src/api';
import { useAppDispatch, useAppSelector } from '@src/models/hooks';
import { updateErrMsg } from '@src/models/app';
import { selectElements, updateElements, updateSelected } from '@src/models/element';
import { selectPipelineId, selectInfo, savePipeline, updateEditing } from '@src/models/pipeline';
import { selectShow, toggle } from '@src/models/setting';
import { updateLoading, selectLoading } from '@src/models/task';
import Setting from './components/Setting';
import NodeType from './components/NodeType';
import style from './style';
import { useTranslation } from 'react-i18next';

const { Item } = Stack;

const EditorBody: React.FC = () => {
  const dispatch = useAppDispatch();
  const elements = useAppSelector(selectElements);
  const pipelineId = useAppSelector(selectPipelineId);
  const pipelineInfo = useAppSelector(selectInfo);
  const settingShow = useAppSelector(selectShow);
  const taskLoading = useAppSelector(selectLoading);
  const reactFlowWrapper = useRef<any>(null);
  const [reactFlowInstance, setReactFlowInstance] = useState<any>(null);
  const { t, i18n } = useTranslation();

  // 加载
  const onLoad = (_reactFlowInstance: OnLoadParams) => {
    setReactFlowInstance(_reactFlowInstance);
  };
  // 点击
  const onElementClick = (_: MouseEvent, element: FlowElement) => {
    if (settingShow && element.data) {
      dispatch(toggle());
    }
  };
  // 增加节点
  const onDrop = (event: any) => {
    event.preventDefault();

    const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
    const modelInfo = JSON.parse(event.dataTransfer.getData('application/reactflow'));
    const position = reactFlowInstance.project({
      x: event.clientX - reactFlowBounds.left,
      y: event.clientY - reactFlowBounds.top,
    });

    const { args } = modelInfo;
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
        });
    }
  };
  // 变更拖拽元素状态
  const onDragOver = (event: any) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  };
  // 拖拽停止事件
  const onNodeDragStop = (_: MouseEvent, node: Node) => {
    dispatch(
      updateElements(
        elements.map((ele: any) => {
          if (ele?.id === node?.id) {
            return node;
          }
          return ele;
        }),
      ),
    );
    dispatch(updateEditing(true));
  };
  // 删除
  const onElementsRemove = (elementsToRemove: Elements) => {
    elementsToRemove.forEach(ele => {
      if (ele?.id && isNode(ele)) {
        api
          .task_modelview_del(+ele.id)
          .then(() => {
            setTimeout(() => {
              dispatch(savePipeline());
            }, 2000);
          })
          .catch(err => {
            if (err.response) {
              dispatch(updateErrMsg({ msg: err.response.data.message }));
            }
          });
      }
    });
    dispatch(updateEditing(true));
    dispatch(updateElements(removeElements(elementsToRemove, elements)));
  };
  // 点与点连线
  const onConnect = (params: Connection | Edge) => {
    dispatch(updateEditing(true));
    dispatch(updateElements(addEdge(Object.assign(params, { arrowHeadType: 'arrow' }), elements)));
  };
  // 选中的元素
  const onSelectionChange = (elements: Elements | null) => {
    if (elements) dispatch(updateSelected(elements));
  };

  // 从接口获取可视化流水线数据
  useEffect(() => {
    const expand = JSON.parse(pipelineInfo?.expand || '[]');
    const elements = expand.map((ele: Node | Edge) => {
      if (isEdge(ele) && !ele?.arrowHeadType) {
        return Object.assign(ele, { arrowHeadType: 'arrow' });
      }
      if (isNode(ele) && !ele.data.label) {
        ele.data.label = ele.data.name;
      }
      return ele;
    });

    dispatch(updateElements(elements));
  }, [pipelineInfo]);

  return (
    <Item shrink grow={1} className={style.itemStyles}>
      <div className={style.flowWrapper} ref={reactFlowWrapper}>
        <ReactFlow
          elements={elements}
          nodeTypes={NodeType}
          snapToGrid={true}
          snapGrid={[16, 16]}
          defaultZoom={1}
          onLoad={onLoad}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onElementClick={onElementClick}
          onElementsRemove={onElementsRemove}
          onConnect={onConnect}
          onNodeDragStop={onNodeDragStop}
          onSelectionChange={onSelectionChange}
        >
          {/* pipeline setting 配置板 */}
          <Setting></Setting>
          <MiniMap nodeStrokeColor="#8a8886" nodeColor="#c8c8c8"></MiniMap>
          <Controls></Controls>
          {/* pipeline task 配置板 */}
          <LayerHost id="hostId_model" className="layer-host" />
        </ReactFlow>
        {/* Loading */}
        <div
          className={style.spinnerWrapper}
          style={{
            visibility: taskLoading ? 'visible' : 'hidden',
          }}
        >
          <Spinner size={SpinnerSize.large} label="Loading"></Spinner>
        </div>
      </div>
    </Item>
  );
};

export default EditorBody;
