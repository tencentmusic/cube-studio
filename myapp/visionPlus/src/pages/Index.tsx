import React from 'react';
import { Stack, IStackStyles } from '@fluentui/react';
import EditorAce from '@src/components/EditorAce';
import FlowEditor from '@src/components/FlowEditor';
import ModuleTree from '@src/components/ModuleTree';
import { useAppSelector } from '@src/models/hooks';
import { selectIsLoadError } from '@src/models/pipeline';
import { useTranslation } from 'react-i18next';

// app 页面初始化样式
const appContainerStyle: IStackStyles = {
  root: {
    width: '100%',
    height: '100%',
    overflow: 'hidden',
  },
};

const App: React.FC = () => {
  const isLoadError = useAppSelector(selectIsLoadError);
  const { t, i18n } = useTranslation();

  return (
    <Stack className="app-container" horizontal styles={appContainerStyle}>
      {
        isLoadError ? <div style={{
          width: '100vw',
          height: '100vh',
          position: 'absolute',
          backgroundColor: 'rgba(0,0,0,0.5)',
          zIndex: 99,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#fff',
          cursor: 'pointer',
          fontSize: 20
        }} onClick={() => {
          window.location.reload()
        }}>
          {t('解析异常，请点击刷新重试')}
        </div> : null
      }
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
