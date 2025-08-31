import React, { useState, useEffect, useRef } from 'react';
import CodeMirror, { ReactCodeMirrorProps } from '@uiw/react-codemirror';
import { json } from '@codemirror/lang-json';
import { EditorView } from '@codemirror/view';
import { useTranslation } from 'react-i18next';
import { Button } from 'antd';

interface IProps extends ReactCodeMirrorProps {
  onChange?: (value: string) => void;
  value?: string;
  rows?: number
}

export default function JsonEditor(props: IProps) {
  const { t } = useTranslation();
  const [isValidJson, setIsValidJson] = useState(true);
  const [rawValue, setRawValue] = useState(props.value || '{}');
  const [editorHeight, setEditorHeight] = useState('auto');
  const editorRef = useRef<HTMLDivElement>(null);
  const isUserEditing = useRef(false);
  const prevPropsValue = useRef(props.value);


  // 格式化函数
  const formatJsonString = (jsonString: string) => {
    try {
      const parsed = JSON.parse(jsonString);
      return JSON.stringify(parsed, null, 2);
    } catch (e) {
      return jsonString;
    }
  };

  // 处理外部value变化
  useEffect(() => {
    if (props.value !== prevPropsValue.current && !isUserEditing.current) {
      const formatted = formatJsonString(props.value || '{}');
      setRawValue(formatted);
      setIsValidJson(true);
      prevPropsValue.current = props.value;
    }
  }, [props.value]);

  // 计算编辑器高度
  const calculateHeight = () => {
    if (!editorRef.current) return;

    const editor = editorRef.current.querySelector('.cm-editor');
    if (!editor) return;

    // 获取内容高度
    const contentHeight = editor.scrollHeight;
    // 设置最大高度为300px
    const height = Math.min(contentHeight, 300);
    setEditorHeight(`${height}px`);
  };

  // 内容变化时重新计算高度
  useEffect(() => {
    calculateHeight();
  }, [rawValue]);


  // 处理编辑器变化
  const handleEditorChange = (value: string) => {
    isUserEditing.current = true;
    setRawValue(value);

    try {
      JSON.parse(value);
      setIsValidJson(true);
      props.onChange?.(value);
    } catch (e) {
      setIsValidJson(false);
      props.onChange?.(value);
    }
  };

  // 手动格式化
  const handleFormatClick = () => {
    const formatted = formatJsonString(rawValue);
    setRawValue(formatted);
    setIsValidJson(true);
    props.onChange?.(formatted);
    isUserEditing.current = false;
  };

  return (
    <div style={{
      position: 'relative'
    }}>
      <CodeMirror
        value={rawValue}
        onChange={handleEditorChange}
        readOnly={props.readOnly}
        height={editorHeight}
        extensions={[
            json(),
            EditorView.lineWrapping,
            EditorView.theme({
            "&": {
              maxHeight: "300px",
            },
            ".cm-scroller": {
              overflow: "auto",
            },
          }),
        ]}
        placeholder={props.placeholder}
        basicSetup={{ lineNumbers: false }}
      />
      <div style={{
        position: 'absolute',
        top: 10,
        right: 10,
        zIndex: 100,
      }}>
        <Button
          size="small"
          type="primary"
          onClick={handleFormatClick}
          disabled={!isValidJson}
        >
          {t("格式化")}
        </Button>
      </div>
      {!isValidJson && (
        <div style={{ color: 'red', marginTop: '10px' }}>
          {t("json格式错误")}
        </div>
      )}
    </div>
  );
}
