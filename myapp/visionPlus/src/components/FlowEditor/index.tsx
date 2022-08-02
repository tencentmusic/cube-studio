import React, { useEffect } from 'react';
import { Stack } from '@fluentui/react';
import { useAppDispatch, useAppSelector } from '@src/models/hooks';
import { getPipeline, selectIsLoadError } from '@src/models/pipeline';
import EditorBody from './components/EditorBody';
import EditorHead from './components/EditorHead';
import EditorTool from './components/EditorTool';

const { Item } = Stack;

const FlowEditor: React.FC = () => {
  const dispatch = useAppDispatch();

  useEffect(() => {
    // 获取 pipeline 信息
    dispatch(getPipeline());
  }, []);

  return (
    <Item
      className="flow-editor"
      styles={{
        root: {
          flexGrow: 1,
          flexShrink: 1,
          background: '#f4f4f4',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
        },
      }}
    >
      <Stack grow={1}>
        {/* 头部信息 */}
        <EditorHead></EditorHead>
        {/* 工具栏 */}
        <EditorTool></EditorTool>
        {/* 流水线交互面板 */}
        <EditorBody></EditorBody>
      </Stack>
    </Item>
  );
};

export default FlowEditor;
