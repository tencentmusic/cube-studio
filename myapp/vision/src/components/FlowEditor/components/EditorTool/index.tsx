import React, { useEffect } from 'react';
import { Stack, CommandBar, ICommandBarItemProps, Icon, Spinner, SpinnerSize } from '@fluentui/react';
import { removeElements, isNode } from 'react-flow-renderer';
import { useAppSelector, useAppDispatch } from '../../../../models/hooks';
import { updateErrMsg } from '../../../../models/app';
import { selectElements, updateElements, selectSelected } from '../../../../models/element';
import {
  selectSaved,
  savePipeline,
  selectInfo,
  selectPipelineId,
  selectEditing,
  updateEditing,
} from '../../../../models/pipeline';
import { saveTaskList, selectTaskList } from '../../../../models/task';
import { toggle } from '../../../../models/template';
import api from '../../../../api';
import style from './style';

const { Item } = Stack;

const EditorTool: React.FC = () => {
  const dispatch = useAppDispatch();
  const elements = useAppSelector(selectElements);
  const saved = useAppSelector(selectSaved);
  const pipeline = useAppSelector(selectInfo);
  const pipelineId = useAppSelector(selectPipelineId);
  const taskList = useAppSelector(selectTaskList);
  const isEditing = useAppSelector(selectEditing);
  const selectedElements = useAppSelector(selectSelected);
  const commandItem: ICommandBarItemProps[] = [
    {
      buttonStyles: style.commonButton,
      iconOnly: true,
      key: 'expand',
      iconProps: {
        iconName: 'Library',
        styles: style.commonIcon,
      },
      text: '展开/关闭菜单',
      onClick: e => {
        e?.preventDefault();
        dispatch(toggle());
      },
    },
    {
      buttonStyles: style.commonButton,
      iconOnly: true,
      key: 'save',
      iconProps: {
        iconName: 'save',
        styles: style.commonIcon,
      },
      text: '保存',
      onClick: async e => {
        e?.preventDefault();
        await dispatch(await saveTaskList());
        dispatch(savePipeline());
      },
    },
    {
      buttonStyles: style.commonButton,
      iconOnly: true,
      key: 'example',
      iconProps: {
        iconName: 'FastForward',
        styles: style.commonIcon,
      },
      text: '运行实例',
      onClick: () => {
        if (pipelineId) {
          window.open(
            `${window.location.origin}/workflow_modelview/list/?_flt_2_labels=%22pipeline-id%22%3A+%22${pipelineId}%22`,
          );
        }
      },
    },
    {
      buttonStyles: style.commonButton,
      iconOnly: true,
      key: 'log',
      iconProps: {
        iconName: 'ComplianceAudit',
        styles: style.commonIcon,
      },
      text: '日志',
      onClick: () => {
        if (pipeline?.id) {
          window.open(`${window.location.origin}/pipeline_modelview/web/log/${pipelineId}`);
        }
      },
    },
    {
      buttonStyles: style.commonButton,
      iconOnly: true,
      key: 'docker',
      iconProps: {
        iconName: 'WebAppBuilderFragment',
        styles: style.commonIcon,
      },
      text: '容器',
      onClick: () => {
        if (pipeline?.name) {
          window.open(`${window.location.origin}/pipeline_modelview/web/pod/${pipelineId}`);
        }
      },
    },
    {
      buttonStyles: style.commonButton,
      iconOnly: true,
      key: 'delete',
      iconProps: {
        iconName: 'Delete',
        styles: style.commonIcon,
      },
      text: '删除节点',
      onClick: () => {
        if (isNode(selectedElements[0])) {
          const taskId = +selectedElements[0]?.id;
          api
            .task_modelview_del(taskId)
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
        dispatch(updateEditing(true));
        dispatch(updateElements(removeElements(selectedElements, elements)));
      },
    },
  ];

  // task 发生编辑行为时状态变更
  useEffect(() => {
    taskList &&
      taskList.forEach((value: any) => {
        if (Object.keys(value).length > 0) {
          dispatch(updateEditing(true));
        }
      });
  }, [taskList]);

  return (
    <Item shrink styles={style.editorToolStyle}>
      <CommandBar id="authoring-page-toolbar" styles={style.commandBarStyle} items={commandItem}></CommandBar>
      <Stack
        horizontal
        verticalAlign="center"
        className={style.autoSavedTips}
        style={{
          visibility: pipelineId ? 'visible' : 'hidden',
        }}
      >
        {saved ? (
          isEditing ? (
            <>
              <Icon iconName="AlertSolid" styles={{ root: { color: '#e95f39', marginRight: 5 } }} />
              未保存
            </>
          ) : (
            <>
              <Icon iconName="SkypeCircleCheck" styles={{ root: { color: '#8cb93c', marginRight: 5 } }} />
              已保存
            </>
          )
        ) : (
          <>
            <Spinner
              styles={{
                root: {
                  marginRight: 5,
                },
              }}
              size={SpinnerSize.small}
            ></Spinner>
            保存中
          </>
        )}
      </Stack>
    </Item>
  );
};

export default EditorTool;
