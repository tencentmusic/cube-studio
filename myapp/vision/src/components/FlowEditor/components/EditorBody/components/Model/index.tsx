import React, { useState, useEffect, FormEvent } from 'react';
import {
  CommandBar,
  ICommandBarItemProps,
  TextField,
  Dropdown,
  IDropdownOption,
  ActionButton,
  Label,
  Toggle,
  Slider,
  SpinButton
} from '@fluentui/react';
import api from '@src/api';
import { updateErrMsg } from '@src/models/app';
import { updateShowEditor, updateKeyValue, selectShowEditor, selectKey, selectValue } from '@src/models/editor';
import { selectElements, updateElements } from '@src/models/element';
import { useAppDispatch, useAppSelector } from '@src/models/hooks';
import { updateTaskList, updateTaskId, selectTaskId } from '@src/models/task';
import style from './style';
import { Switch } from 'antd';
import { useTranslation } from 'react-i18next';

interface ModelProps {
  model: any;
}
const Model: React.FC<ModelProps> = props => {
  const dispatch = useAppDispatch();
  const elements = useAppSelector(selectElements);
  const taskId = useAppSelector(selectTaskId);
  const jsonEditorShow = useAppSelector(selectShowEditor);
  const jsonEditorKey = useAppSelector(selectKey);
  const jsonEditorValue = useAppSelector(selectValue);
  const [task, setTask] = useState<any>({});
  const [taskChanged, setTaskChanged] = useState<any>({});
  const [jobTemplate, setJobTemplate] = useState<any>({});
  const [templateArgs, setTemplateArgs] = useState<any>({});
  const [taskArgs, setTaskArgs] = useState<any>({});
  const { t, i18n } = useTranslation();
  const [message, setMessage] = useState('clear');
  const _overflowItems: ICommandBarItemProps[] = [
    {
      key: 'debug',
      // iconOnly: true,
      text: 'debug',
      onClick: () => handleTaskEvent('debug'),
      iconProps: { iconName: 'Bug' },
    },
    {
      key: 'run',
      text: 'run',
      // iconOnly: true,
      onClick: () => handleTaskEvent('run'),
      iconProps: { iconName: 'Play' },
    },
    {
      key: 'log',
      text: 'log',
      // iconOnly: true,
      onClick: () => handleTaskEvent('log'),
      iconProps: { iconName: 'TimeEntry' },
    },
    {
      key: 'clear',
      text: message || 'clear',
      // iconOnly: true,
      onClick: () => handleTaskClearEvent('clear'),
      iconProps: { iconName: 'Unsubscribe' },
    },
  ];

  // 处理 task 跳转事件
  const handleTaskEvent = (type: string) => {
    setMessage('clear');
    if (props.model.id) {
      window.open(`${window.location.origin}/task_modelview/api/${type}/${props.model.id}`);
    }
  };
  // 处理 task 清理 事件
  const handleTaskClearEvent = (type: string) => {
    if (props.model.id) {
      setMessage('cleaning');
      api.task_modelview_clear(props.model.id).then((res: any) => {
        setMessage('cleared');
      })
      // window.open(`${window.location.origin}/task_modelview/api/${type}/${props.model.id}`);
    }
  };
  // 配置变化事件
  const handleOnChange = (key: string, value: string | number | boolean | object, type?: string) => {
    const obj: any = {};
    let res = null;
    // console.log(key)
    // console.log(value)
    switch (type) {
      case 'json':
        try {
          res = JSON.parse(`${value}`);
        } catch (error) {
          res = value;
        }
        break;
      case 'int':
        res = +value;
        break;
      case 'float':
        res = +value;
        break;
      default:
        res = value;
        break;
    }

    if (type) {
      const args = JSON.parse(JSON.stringify(taskArgs));
      args[key] = res;
      obj.args = JSON.stringify(args);
    } else {
      obj[key] = value;
    }

    setTaskChanged({ ...taskChanged, ...obj });
    setTask({ ...task, ...obj });
    if (obj?.args) {
      setTaskArgs({ ...taskArgs, ...JSON.parse(obj.args) });
    }
  };

  // 从接口获取数据展示 task 配置面板
  useEffect(() => {
    if (props.model.selected) {
      dispatch(updateTaskId(+props.model.id));
      if (Object.keys(task).length === 0) {
        api
          .task_modelview_get(+props.model.id)
          .then((res: any) => {
            if (res.status === 0) {
              const taskArgs = JSON.parse(res.result.args);
              const jobTemplate = res.result.job_template;
              const args = jobTemplate?.args ? JSON.parse(jobTemplate.args) : {};
              const initArgs = Object.keys(args).reduce((acc: any, cur: string) => {
                const current = args[cur];

                Object.keys(current).forEach((key: string) => {
                  acc[key] = current[key].default; // 参数的默认值
                });

                return acc;
              }, {});

              setTask(res.result);
              setTaskArgs(Object.assign(initArgs, taskArgs));
              setTemplateArgs(args);
              setJobTemplate(jobTemplate);
            }
          })
          .catch(err => {
            if (err.response) {
              dispatch(updateErrMsg({ msg: err.response.data.message }));
            }
          });
      }
    }
  }, [props.model.selected]);

  // 将变化的 task 配置同步至 redux
  useEffect(() => {
    if (props.model.id) {
      dispatch(
        updateTaskList({
          id: +props.model.id,
          changed: taskChanged,
        }),
      );
    }
    if (taskChanged?.label) {
      const res = elements.map(ele => {
        if (ele.id === props.model.id) {
          const data = { ...ele.data, ...{ label: taskChanged.label } };
          return { ...ele, ...{ data } };
        }
        return ele;
      });
      dispatch(updateElements(res));
    }
  }, [taskChanged]);

  // json 数据编辑
  useEffect(() => {
    if (Object.keys(templateArgs).length > 0 && +props.model.id === taskId && !jsonEditorShow) {
      handleOnChange(jsonEditorKey, jsonEditorValue, 'json');
    }
  }, [jsonEditorValue]);

  return (
    <div
      className={style.modelContainer}
      style={{
        visibility: props.model.selected ? 'visible' : 'hidden',
      }}
    >
      <div className={style.modelHeader}>
        <div className={style.headerTitle}>{task?.name || ''}</div>
      </div>
      {/* task 配置 */}
      <div className={style.modelContent}>
        <div className={style.contentWrapper}>
          {/* task 操作 */}
          <CommandBar
            items={_overflowItems}
            overflowItems={[]}
            styles={{
              root: {
                padding: 0,
              },
            }}
          />
          <TextField
            label={t('任务模板')}
            value={jobTemplate?.name || ''}
            disabled
            readOnly
            onRenderDescription={() => {
              let help_url = '';
              try {
                if (jobTemplate.expand) {
                  help_url = JSON.parse(jobTemplate.expand)?.help_url || '';
                }
              } catch (error) {
                console.error(error);
              }
              return (
                <div className={style.templateConfig}>
                  {help_url ? (
                    <a href={help_url} target="_blank" rel="noreferrer">
                      {t('配置文档')}
                    </a>
                  ) : null}
                </div>
              );
            }}
          />
          <div className={style.splitLine}></div>
          <TextField label={t('模板描述')} value={jobTemplate?.describe || ''} disabled readOnly />
          <div className={style.splitLine}></div>
          <TextField
            label={t('名称')}
            onChange={(event: FormEvent, value?: string) => {
              handleOnChange('name', value ? value : '');
            }}
            value={task?.name || ''}
            disabled
          />
          <div className={style.splitLine}></div>
          <TextField
            label={t('标签')}
            description={t('节点标签')}
            required
            onChange={(event: FormEvent, value?: string) => {
              handleOnChange('label', value ? value : '');
            }}
            value={task?.label || ''}
          />
          <div className={style.splitLine}></div>
          <TextField
            label={t('内存申请')}
            description={t('内存的资源使用限制，示例1G，10G， 最大100G，如需更多联系管理员')}
            onChange={(event: FormEvent, value?: string) => {
              handleOnChange('resource_memory', value ? value : '');
            }}
            value={task?.resource_memory || ''}
            required
          />
          <div className={style.splitLine}></div>
          <TextField
            label={t('CPU申请')}
            description={t('CPU的资源使用限制(单位核)，示例 0.4，10，最大50核，如需更多联系管理员')}
            onChange={(event: FormEvent, value?: string) => {
              handleOnChange('resource_cpu', value ? value : '');
            }}
            value={task?.resource_cpu || ''}
            required
          />
          <div className={style.splitLine}></div>
          <TextField
            label={t('GPU申请')}
            description={t('gpu的资源使用限制(单位卡)，示例:1，2，训练任务每个容器独占整卡。申请具体的卡型号，可以类似 1(V100)')}
            onChange={(event: FormEvent, value?: string) => {
              handleOnChange('resource_gpu', value ? value : '');
            }}
            value={task?.resource_gpu || ''}
          />
          <div className={style.splitLine}></div>
          <TextField
            label={t('RDMA申请')}
            description={t('RDMA的资源使用限制，示例 0，1，10，填写方式咨询管理员')}
            onChange={(event: FormEvent, value?: string) => {
              handleOnChange('resource_rdma', value ? value : '');
            }}
            value={task?.resource_rdma || ''}
            required
          />
          <div className={style.splitLine}></div>
          <TextField
            label={t('超时中断')}
            description={t('task运行时长限制，为0表示不限制(单位s)')}
            onChange={(event: FormEvent, value?: string) => {
              handleOnChange('timeout', value ? value : '');
            }}
            value={task?.timeout || 0}
          />
          <div className={style.splitLine}></div>
          <TextField
            label={t('重试次数')}
            description={t('task重试次数')}
            onChange={(event: FormEvent, value?: string) => {
              handleOnChange('retry', value ? value : '');
            }}
            value={task?.retry || 0}
          />
          <div className={style.splitLine}></div>

          <div style={{ fontWeight: 600, padding: '5px 0px' }}>{t('是否跳过')}</div>

          <Switch checkedChildren={t('是')} unCheckedChildren={t('否')} checked={!!task?.skip} onChange={(checked) => {
            handleOnChange('skip', checked);
          }} />

          {/* 模板的参数动态渲染 */}
          {Object.keys(templateArgs).reduce((acc, cur) => {
            const current = templateArgs[cur];

            const mapCurrent = Object.keys(current).map(key => {
              const args = current[key];
              const { choice } = args;
              const options = choice.map((option: string) => {
                if(typeof option === 'string') {
                  return {
                    key: option,
                    text: option,
                  };
                }else{
                  return {
                    key: option['key'],
                    text: option['text'],
                  };
                }
              });
              console.log(options)
              const keyArgs = taskArgs && taskArgs[key];
              const keyValue = args.type === 'json' ? JSON.stringify(keyArgs, undefined, 4) : keyArgs;

              if(args.type==='float'){
                const range = typeof args.range === 'string' ? args.range.split(',') : args.range;
                return (<React.Fragment key={key}>
                  {
                    <>
                    <Slider
                        styles={{ valueLabel: { margin: '0', width: 'auto' } }}
                        label={`${key}`}
                        min={range[0] || 0}
                        max={range[1] || 1}
                        step= {args.step || 0.1 }
                        onChange={(value?: number) => {
                          handleOnChange(key, value ? value : 0, args.type);
                        }}
                        value={keyValue}
                        disabled={args.editable !== 1}
                        showValue
                      />
                      <div className={style.argsDescription} dangerouslySetInnerHTML={{ __html: args.describe }}></div>
                    </>
                  }
                  </React.Fragment>
                )
              }
              if(args.type==='int'){
                const range = typeof args.range === 'string' ? args.range.split(',') : args.range;
                return (<React.Fragment key={key}>
                  {
                    <div style={{ width: '100%' }}>
                      <SpinButton
                          styles={
                            {
                                root: {
                                  display: 'flex',
                                  flexDirection: 'column',
                                  alignItems: 'flex-start',
                                  width: "100%"
                                },
                                spinButtonWrapper: {
                                  width: '100%',
                                },
                                labelWrapper: {
                                  marginBottom: '4px', // 调整标签与输入框之间的间距
                                },
                              }
                          }
                          label={`${key}`}
                          min={range || 0}
                          max={range || 100}
                          step= {args.step || 1 }
                          onIncrement={(value?: string) => {
                            handleOnChange(key, value ? parseInt(value,10)+1 : 1, args.type);
                          }}
                          onDecrement={(value?: string) => {
                            handleOnChange(key, value ? parseInt(value,10)-1 : -1, args.type);
                          }}
                          value={keyValue}
                          disabled={args.editable !== 1}
                        />
                        <div className={style.argsDescription} dangerouslySetInnerHTML={{ __html: args.describe }}></div>

                    </div>

                  }
                  </React.Fragment>
                )
              }
              if(args.type==='bool'){
                return (<React.Fragment key={key}>
                  {
                    <>
                    <Toggle
                        label={`${key}`}
                        checked={keyValue}
                        onChange={(event: FormEvent, value?: boolean) => {
                          handleOnChange(key, value ? value : false, args.type);
                        }}
                        onText="On"
                        offText="Off"
                        disabled={args.editable !== 1}

                      />
                      <div className={style.argsDescription} dangerouslySetInnerHTML={{ __html: args.describe }}></div>
                    </>
                  }
                  </React.Fragment>
                )
              }

              if(args.type==='list'){
                const selectedKeys = (typeof keyValue === 'string' ? keyValue : args.default).split(',');

                return (<React.Fragment key={key}>
                  {
                    <>
                      <Dropdown
                        label={`${key}`}
                        onChange={(event: FormEvent, option?: IDropdownOption) => {
                          let currentSelectedKeys =  (typeof keyValue === 'string' ? keyValue : args.default).split(',');
                          // 去除空白字符串
                          currentSelectedKeys = currentSelectedKeys.filter((str: string) => str.trim() !== '');
                          let newSelectedKeys=currentSelectedKeys
                          if(option?.selected){
                            // 去重
                            if(!currentSelectedKeys.includes(option.key)){
                              newSelectedKeys = [...currentSelectedKeys, option.key as string]
                            }
                          }else{
                             newSelectedKeys = currentSelectedKeys.filter((key: string)=> key !== option?.key)
                          }
                          const newSelectedKeys_str = newSelectedKeys?.join(',');
                          handleOnChange(key, newSelectedKeys_str || '', args.type);
                        }}
                        selectedKeys={selectedKeys}  // 这里有bug，无效
                        options={options}
                        required={args.require === 1}
                        disabled={args.editable !== 1}
                        multiSelect
                      />
                      <div className={style.argsDescription} dangerouslySetInnerHTML={{ __html: args.describe }}></div>
                    </>
                  }
                  </React.Fragment>
                )
              }

              return (
                <React.Fragment key={key}>
                  {options.length > 0 ? (
                    <>
                      <Dropdown
                        label={`${key}`}
                        onChange={(event: FormEvent, option?: IDropdownOption) => {
                          handleOnChange(key, `${option?.key}` || '', args.type);
                        }}
                        defaultSelectedKey={keyValue || args.default}
                        options={options}
                        required={args.require === 1}
                        disabled={args.editable !== 1}
                      />
                      <div className={style.argsDescription} dangerouslySetInnerHTML={{ __html: args.describe }}></div>
                    </>
                  ) : (
                      <TextField
                        onRenderLabel={() => {
                          return (
                            <div className={style.textLabelStyle}>
                              {`${key}`}
                              {args.type === 'json' || args.type === 'text' ? (
                                <ActionButton
                                  iconProps={{ iconName: 'FullWidthEdit' }}
                                  onClick={() => {
                                    dispatch(
                                      updateKeyValue({
                                        key,
                                        value: keyValue,
                                        type: args.type ==='json'?'json':(args.item_type || 'str')
                                      }),
                                    );
                                    dispatch(updateShowEditor(true));
                                  }}
                                >
                                  {t('编辑')}
                                </ActionButton>
                              ) : null}
                            </div>
                          );
                        }}
                        onRenderDescription={() => {
                          return (
                            <div
                              className={style.argsDescription}
                              dangerouslySetInnerHTML={{ __html: args.describe }}
                            ></div>
                          );
                        }}
                        multiline={args.type === 'json' || args.type === 'text'}
                        autoAdjustHeight={args.type === 'json' || args.type === 'text'}
                        onChange={(event: FormEvent, value?: string) => {
                          handleOnChange(key, value ? value : '', args.type);
                        }}
                        value={keyValue}
                        required={args.require === 1}
                        disabled={args.editable !== 1 || args.type === 'json'}
                      />
                    )}
                </React.Fragment>
              );
            });

            acc.push(
              (
                <React.Fragment key={cur}>
                  <Label>{t('参数')} {cur}</Label>
                  {mapCurrent.flat()}
                  <div className={style.splitLine}></div>
                </React.Fragment>
              ) as never,
            );
            return acc;
          }, [])}
        </div>
      </div>
    </div>
  );
};

export default Model;
