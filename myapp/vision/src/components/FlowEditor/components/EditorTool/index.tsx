import { CommandBar, ICommandBarItemProps, Icon, Spinner, SpinnerSize, Stack } from '@fluentui/react';
import api from '@src/api';
import { updateErrMsg } from '@src/models/app';
import { selectElements, selectSelected, updateElements } from '@src/models/element';
import { useAppDispatch, useAppSelector } from '@src/models/hooks';
import {
  savePipeline,
  selectEditing,
  selectInfo,
  selectPipelineId,
  selectSaved,
  updateEditing,
} from '@src/models/pipeline';
import { saveTaskList, selectTaskList } from '@src/models/task';
import { toggle } from '@src/models/template';
import React, { useEffect } from 'react';
import { isNode, removeElements } from 'react-flow-renderer';
import { useTranslation } from 'react-i18next';
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
  const { t, i18n } = useTranslation();

  const commandItem: ICommandBarItemProps[] = [
    {
      // buttonStyles: style.commonButton,
      // iconOnly: true,
      key: 'expand',
      iconProps: {
        iconName: 'Library',
        styles: style.commonIcon,
      },
      text: t('展开/关闭菜单'),
      onClick: e => {
        e?.preventDefault();
        dispatch(toggle());
      },
    },
    {
      // buttonStyles: style.commonButton,
      // iconOnly: true,
      key: 'save',
      iconProps: {
        iconName: 'save',
        styles: style.commonIcon,
      },
      text: t('保存'),
      onClick: async e => {
        e?.preventDefault();
        dispatch(await saveTaskList());
        dispatch(savePipeline());
      },
    },
    {
      // buttonStyles: style.commonButton,
      // iconOnly: true,
      key: 'example',
      iconProps: {
        iconName: 'FastForward',
        styles: style.commonIcon,
      },
      text: t('调度实例'),
      onClick: () => {
        if (pipelineId) {
          window.open(
            `${window.location.origin}/pipeline_modelview/api/web/workflow/${pipelineId}`,
          );
        }
      },
    },
    {
      // buttonStyles: style.commonButton,
      // iconOnly: true,
      key: 'log',
      iconProps: {
        iconName: 'ComplianceAudit',
        styles: style.commonIcon,
      },
      text: t('日志'),
      onClick: () => {
        if (pipeline?.id) {
          window.open(`${window.location.origin}/pipeline_modelview/api/web/log/${pipelineId}`);
        }
      },
    },
    {
      // buttonStyles: style.commonButton,
      // iconOnly: true,
      key: 'docker',
      iconProps: {
        iconName: 'WebAppBuilderFragment',
        styles: style.commonIcon,
      },
      text: t('容器'),
      onClick: () => {
        if (pipeline?.name) {
          window.open(`${window.location.origin}/pipeline_modelview/api/web/pod/${pipelineId}`);
        }
      },
    },
    {
      // buttonStyles: style.commonButton,
      // iconOnly: true,
      key: 'timer',
      iconProps: {
        iconName: 'TimeEntry',
        styles: style.commonIcon,
      },
      text: t('定时记录'),
      onClick: () => {
        if (pipeline?.name) {
          window.open(`${window.location.origin}/pipeline_modelview/api/web/runhistory/${pipelineId}`);
        }
      },
    },
    {
      // buttonStyles: style.commonButton,
      // iconOnly: true,
      key: 'delete',
      iconProps: {
        iconName: 'Delete',
        styles: style.commonIcon,
      },
      text: t('删除节点'),
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
    {
      // buttonStyles: style.commonButton,
      // iconOnly: true,
      key: 'monitor',
      iconProps: {
        iconName: 'NetworkTower',
        styles: style.commonIcon,
      },
      text: t('监控'),
      onClick: () => {
        if (pipeline?.id) {
          window.open(`${window.location.origin}/pipeline_modelview/api/web/monitoring/${pipelineId}`);
        }
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
          <>
            <Icon
              iconName={isEditing ? 'AlertSolid' : 'SkypeCircleCheck'}
              styles={{ root: { color: isEditing ? '#e95f39' : '#8cb93c', marginRight: 5 } }}
            />
            {isEditing ? t('未保存') : t('已保存')}
          </>
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
            {t('保存中')}
          </>
          )}
      </Stack>
    </Item>
  );
};

export default EditorTool;
