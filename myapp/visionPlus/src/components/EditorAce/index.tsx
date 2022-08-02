import React, { useState, useEffect } from 'react';
import { Icon, IconButton, PrimaryButton, DefaultButton, Stack, MessageBar, MessageBarType } from '@fluentui/react';
import MonacoEditor from 'react-monaco-editor';
import { useAppDispatch, useAppSelector } from '@src/models/hooks';
import { selectShowEditor, updateShowEditor, selectValue, updateValue, selectType } from '@src/models/editor';
import { isJsonString } from '@src/utils/index';
import style from './style';
interface IProps {
  isStrictJson?: boolean
  language?: string
}

const EditorAce: React.FC<IProps> = (props: IProps) => {
  const dispatch = useAppDispatch();
  const showModal = useAppSelector(selectShowEditor);
  const value = useAppSelector(selectValue);
  const type = useAppSelector(selectType);
  const [current, setCurrent] = useState('');
  const [codeWidth, setCodeWidth] = useState<string | number>('100%');
  const [dragging, setDragging] = useState(false);
  const [isError, setIsError] = useState(false);

  const handleConfirm = () => {
    if (props.isStrictJson && !isJsonString(current)) {
      setIsError(true);
      return;
    }
    dispatch(updateValue(current));
    dispatch(updateShowEditor(false));
  };
  const handleCancel = () => {
    setIsError(false);
    dispatch(updateShowEditor(false));
  };
  const onMouseDown = () => {
    setDragging(true);
  };
  const onMouseMove = (event: React.MouseEvent) => {
    if (dragging && event.clientX > 370) {
      setCodeWidth(event.clientX - 270);
    }
  };
  const onMouseUpOrLeave = () => {
    setDragging(false);
  };

  useEffect(() => {
    setCurrent(value);
  }, [showModal]);

  return (
    <div
      className={style.modalStyles}
      style={{
        display: showModal ? 'flex' : 'none',
      }}
    >
      <Stack className={style.contentStyles.container}>
        <div className={style.contentStyles.header}>
          <span>编辑</span>
          <IconButton iconProps={style.cancelIcon} onClick={handleCancel} />
        </div>
        <div
          className={style.contentStyles.body}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUpOrLeave}
          onMouseLeave={onMouseUpOrLeave}
        >
          <div
            style={{
              height: '100%',
              width: codeWidth,
            }}
          >
            <MonacoEditor
              language={type || 'json'}
              // language={'mysql'}
              theme="vs"
              value={current}
              onChange={value => {
                setIsError(false);
                setCurrent(value);
              }}
              height={500}
              options={{
                renderValidationDecorations: 'on',
                automaticLayout: true,
                // formatOnPaste: true,
                // formatOnType: true,
                // renderLineHighlight: 'none',
                autoClosingOvertype: 'always',
                cursorStyle: 'block',
                quickSuggestions: false,
                scrollBeyondLastLine: false,
                snippetSuggestions: 'none',
                minimap: {
                  enabled: false,
                },
              }}
            />
          </div>
          <div className={style.resizeLine} onMouseDown={onMouseDown}>
            <Icon iconName="BulletedListBulletMirrored" className={style.resizeLineIconStyles}></Icon>
          </div>
          <div
            style={{
              flex: 1,
            }}
          >
            {/* <VisualizedData /> */}
          </div>
        </div>
        <div className={style.contentStyles.footer}>
          {isError ? (
            <MessageBar
              messageBarType={MessageBarType.error}
              styles={{
                root: {
                  position: 'absolute',
                  width: '50%',
                  left: 0,
                },
              }}
              isMultiline={false}
            >
              格式错误
            </MessageBar>
          ) : null}
          <PrimaryButton styles={{ root: { marginRight: 10 } }} onClick={handleConfirm}>
            确认
          </PrimaryButton>
          <DefaultButton onClick={handleCancel}>取消</DefaultButton>
        </div>
      </Stack>
    </div>
  );
};

export default EditorAce;
