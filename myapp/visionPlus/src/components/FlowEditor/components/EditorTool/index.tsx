import { AppstoreOutlined, DeleteOutlined, SaveOutlined } from '@ant-design/icons';
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
import { Button, message } from 'antd';
import React, { useEffect, useState } from 'react';
import { isNode, removeElements } from 'react-flow-renderer';
import { useTranslation } from 'react-i18next';
import style from './style';

const { Item } = Stack;

const EditorTool: React.FC = () => {
  const dispatch = useAppDispatch();
  const elements = useAppSelector(selectElements);
  const saved = useAppSelector(selectSaved);
  const info = useAppSelector(selectInfo);
  const pipelineId = useAppSelector(selectPipelineId);
  const taskList = useAppSelector(selectTaskList);
  const isEditing = useAppSelector(selectEditing);
  const selectedElements = useAppSelector(selectSelected);
  const { t, i18n } = useTranslation();

  const [commandList, setCommandList] = useState<any[]>([])

  useEffect(() => {
    if (info) {
      const config = (info?.pipeline_jump_button || []).map((item: any) => {
        const target = (<Button className={style.commandButtonStyle} type="text" key={Math.random().toString(36).substring(2)} onClick={() => {
          window.open(`${window.location.origin}${item.action_url}`);
        }}>
          <div className={style.btnIcon}><span dangerouslySetInnerHTML={{ __html: item.icon_svg }}></span><span>{item.name}</span></div>
        </Button>)
        return target
      })
      setCommandList(config)
    }
  }, [info])

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
      {/* <CommandBar id="authoring-page-toolbar" styles={style.commandBarStyle} items={commandList}></CommandBar> */}
      <div className={style.commandBarStyleCustom}>
        <Button className={style.commandButtonStyle} type="text" icon={<AppstoreOutlined className={style.commonIcon}/>} onClick={() => {
          dispatch(toggle());
        }}>{t('展开/关闭菜单')}</Button>
        <Button className={style.commandButtonStyle} type="text" icon={<SaveOutlined className={style.commonIcon}/>} onClick={() => {
          message.success(t('保存成功'))
          dispatch(savePipeline());
        }}>{t('保存')}</Button>
        <Button className={style.commandButtonStyle} type="text" icon={<DeleteOutlined className={style.commonIcon}/>} onClick={() => {
          if (isNode(selectedElements[0])) {
            dispatch(savePipeline());
          }
          dispatch(updateEditing(true));
          dispatch(updateElements(removeElements(selectedElements, elements)));
        }}>{t('删除节点')}</Button>
        {commandList}
      </div>
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
