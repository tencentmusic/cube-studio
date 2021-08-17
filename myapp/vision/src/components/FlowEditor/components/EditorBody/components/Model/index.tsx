import React, { useState, useEffect, FormEvent } from 'react';
import { CommandBar, ICommandBarItemProps, TextField, Dropdown, IDropdownOption, ActionButton } from '@fluentui/react';
import api from '@src/api';
import { updateErrMsg } from '@src/models/app';
import { updateShowEditor, updateKeyValue, selectShowEditor, selectKey, selectValue } from '@src/models/editor';
import { selectElements, updateElements } from '@src/models/element';
import { useAppDispatch, useAppSelector } from '@src/models/hooks';
import { updateTaskList, updateTaskId, selectTaskId } from '@src/models/task';
import style from './style';

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
  const _overflowItems: ICommandBarItemProps[] = [
    {
      key: 'debug',
      iconOnly: true,
      text: 'task debug',
      onClick: () => handleTaskEvent('debug'),
      iconProps: { iconName: 'Bug' },
    },
    {
      key: 'run',
      text: 'task run',
      iconOnly: true,
      onClick: () => handleTaskEvent('run'),
      iconProps: { iconName: 'Play' },
    },
    {
      key: 'log',
      text: 'task log',
      iconOnly: true,
      onClick: () => handleTaskEvent('log'),
      iconProps: { iconName: 'TimeEntry' },
    },
    {
      key: 'clear',
      text: 'task clear',
      iconOnly: true,
      onClick: () => handleTaskEvent('clear'),
      iconProps: { iconName: 'Unsubscribe' },
    },
  ];

  // 处理 task 跳转事件
  const handleTaskEvent = (type: string) => {
    if (props.model.id) {
      window.open(`${window.location.origin}/task_modelview/${type}/${props.model.id}`);
    }
  };
  // 配置变化事件
  const handleOnChange = (key: string, value: string | number | boolean, type?: string) => {
    const obj: any = {};
    let res = null;

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
      console.log(+props.model.id);
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
    if (taskChanged?.name) {
      const res = elements.map(ele => {
        if (ele.id === props.model.id) {
          const data = { ...ele.data, ...{ name: taskChanged.name } };
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
            label="任务模板"
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
                      配置文档
                    </a>
                  ) : null}
                </div>
              );
            }}
          />
          <div className={style.splitLine}></div>
          <TextField label="模板描述" value={jobTemplate?.describe || ''} disabled readOnly />
          <div className={style.splitLine}></div>
          <TextField
            label="名称"
            description="英文名(字母、数字、- 组成)，最长50个字符"
            onChange={(event: FormEvent, value?: string) => {
              handleOnChange('name', value ? value : '');
            }}
            value={task?.name || ''}
          />
          <div className={style.splitLine}></div>
          <TextField
            label="标签"
            description="中文名"
            required
            onChange={(event: FormEvent, value?: string) => {
              handleOnChange('label', value ? value : '');
            }}
            value={task?.label || ''}
          />
          <div className={style.splitLine}></div>
          <TextField
            label="挂载目录"
            description="外部挂载，格式:$pvc_name1(pvc):/$container_path1,$hostpath1(hostpath):/$container_path2,注意pvc会自动挂载对应目录下的个人rtx子目录"
            onChange={(event: FormEvent, value?: string) => {
              handleOnChange('volume_mount', value ? value : '');
            }}
            value={task?.volume_mount || ''}
          />
          <div className={style.splitLine}></div>
          <TextField
            label="启动目录"
            description="工作目录，容器启动的初始所在目录，不填默认使用Dockerfile内定义的工作目录"
            onChange={(event: FormEvent, value?: string) => {
              handleOnChange('working_dir', value ? value : '');
            }}
            value={task?.working_dir || ''}
          />
          <div className={style.splitLine}></div>
          <TextField
            label="启动命令"
            description="启动命令"
            onChange={(event: FormEvent, value?: string) => {
              handleOnChange('command', value ? value : '');
            }}
            multiline={true}
            autoAdjustHeight={true}
            value={task?.command || ''}
          />
          <div className={style.splitLine}></div>
          <TextField
            label="调度机器"
            description="运行当前task所在的机器"
            onChange={(event: FormEvent, value?: string) => {
              handleOnChange('node_selector', value ? value : '');
            }}
            value={task?.node_selector || ''}
          />
          <div className={style.splitLine}></div>
          <TextField
            label="内存申请"
            description="内存的资源使用限制，示例1G，10G， 最大10G，如需更多联系管理员"
            onChange={(event: FormEvent, value?: string) => {
              handleOnChange('resource_memory', value ? value : '');
            }}
            value={task?.resource_memory || ''}
            required
          />
          <div className={style.splitLine}></div>
          <TextField
            label="CPU申请"
            description="CPU的资源使用限制(单位核)，示例 0.4，10，最大10核，如需更多联系管理员"
            onChange={(event: FormEvent, value?: string) => {
              handleOnChange('resource_cpu', value ? value : '');
            }}
            value={task?.resource_cpu || ''}
            required
          />
          <div className={style.splitLine}></div>
          <TextField
            label="GPU申请"
            description="GPU的资源使用限制(单位卡)，示例:1，2，训练任务每个容器独占整卡"
            onChange={(event: FormEvent, value?: string) => {
              handleOnChange('resource_gpu', value ? value : '');
            }}
            value={task?.resource_gpu || ''}
          />
          <div className={style.splitLine}></div>
          <TextField
            label="超时中断"
            description="task运行时长限制，为0表示不限制(单位s)"
            onChange={(event: FormEvent, value?: string) => {
              handleOnChange('timeout', value ? value : '');
            }}
            value={task?.timeout || 0}
          />
          <div className={style.splitLine}></div>
          <TextField
            label="重试次数"
            description="task重试次数"
            onChange={(event: FormEvent, value?: string) => {
              handleOnChange('retry', value ? value : '');
            }}
            value={task?.retry || 0}
          />
          <div className={style.splitLine}></div>
          {/* 模板的参数动态渲染 */}
          {Object.keys(templateArgs).reduce((acc, cur) => {
            const current = templateArgs[cur];

            const mapCurrent = Object.keys(current).map(key => {
              const args = current[key];
              const { choice } = args;
              const options = choice.map((option: string) => {
                return {
                  key: option,
                  text: option,
                };
              });
              const keyArgs = taskArgs && taskArgs[key];
              const keyValue = args.type === 'json' ? JSON.stringify(keyArgs, undefined, 4) : keyArgs;

              return (
                <React.Fragment key={key}>
                  {options.length > 0 ? (
                    <>
                      <Dropdown
                        label={`${cur} ${key}`}
                        onChange={(event: FormEvent, option?: IDropdownOption) => {
                          handleOnChange(key, `${option?.text}` || '', args.type);
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
                            {`${cur} ${key}`}
                            {args.type === 'json' ? (
                              <ActionButton
                                iconProps={{ iconName: 'FullWidthEdit' }}
                                onClick={() => {
                                  dispatch(
                                    updateKeyValue({
                                      key,
                                      value: keyValue,
                                    }),
                                  );
                                  dispatch(updateShowEditor(true));
                                }}
                              >
                                编辑
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
                      multiline={args.type === 'json'}
                      autoAdjustHeight={args.type === 'json'}
                      onChange={(event: FormEvent, value?: string) => {
                        handleOnChange(key, value ? value : '', args.type);
                      }}
                      value={keyValue}
                      required={args.require === 1}
                      disabled={args.editable !== 1 || args.type === 'json'}
                    />
                  )}
                  <div className={style.splitLine}></div>
                </React.Fragment>
              );
            });

            acc.push(mapCurrent.flat() as never);
            return acc;
          }, [])}
        </div>
      </div>
    </div>
  );
};

export default Model;
