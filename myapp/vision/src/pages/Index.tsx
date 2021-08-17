import React from 'react';
import { Stack, IStackStyles } from '@fluentui/react';
import EditorAce from '@src/components/EditorAce';
import FlowEditor from '@src/components/FlowEditor';
import ModuleTree from '@src/components/ModuleTree';

// app 页面初始化样式
const appContainerStyle: IStackStyles = {
  root: {
    width: '100%',
    height: '100%',
    overflow: 'hidden',
  },
};

const App: React.FC = () => {
  return (
    <Stack className="app-container" horizontal styles={appContainerStyle}>
      {/* 任务模板库 */}
      <ModuleTree />
      {/* 流水线编辑 */}
      <FlowEditor />
      {/* JSON 编辑器 */}
      <EditorAce />
    </Stack>
  );
};

export default App;
