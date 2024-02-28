import React, { useState, useEffect, FormEvent } from 'react';
import { IconButton, Dropdown, IDropdownOption, TextField } from '@fluentui/react';
import { isEdge } from 'react-flow-renderer';
import api from '@src/api';
import { useAppDispatch, useAppSelector } from '@src/models/hooks';
import { selectElements } from '@src/models/element';
import { selectInfo, updateChanged, updateEditing, selectChanged } from '@src/models/pipeline';
import { selectShow, toggle } from '@src/models/setting';
import style from './style';
import { DatePicker, Select, Switch } from 'antd';
import 'moment/locale/zh-cn';
import locale from 'antd/es/date-picker/locale/zh_CN';
import moment from 'moment';
import { useTranslation } from 'react-i18next';

const Setting: React.FC = () => {
  const dispatch = useAppDispatch();
  const pipelineInfo = useAppSelector(selectInfo);
  const settingShow = useAppSelector(selectShow);
  const pipelineChanged = useAppSelector(selectChanged);
  const elements = useAppSelector(selectElements);
  const [current, setCurrent] = useState<any>({});
  const [dropItem, setDropItem] = useState([]);
  const [selectedItem, setSelectedItem] = useState<IDropdownOption>();
  const { t, i18n } = useTranslation();

  // 配置变化事件
  const handleOnChange = (key: string, value: string | number | boolean | IDropdownOption) => {
    const obj: any = {};
    obj[key] = value;
    if (key === 'project') {
      const item = value as IDropdownOption;
      setSelectedItem(item);
      obj[key] = item.key;
    }
    if (pipelineInfo) {
      setCurrent({ ...current, ...obj });
      dispatch(updateChanged({ ...pipelineChanged, ...obj }));
    }
  };

  // 初始化选项
  useEffect(() => {
    if (pipelineInfo) {
      setCurrent(pipelineInfo);
      const selected: IDropdownOption = {
        key: pipelineInfo?.project?.id,
        text: pipelineInfo?.project?.name,
      };
      setSelectedItem(selected);
    }
  }, [pipelineInfo]);

  // 将 setting 编辑变化的数据同步至 redux
  useEffect(() => {
    if (Object.keys(pipelineChanged).length > 1) {
      dispatch(updateEditing(true));
    }
  }, [pipelineChanged]);

  // 项目组选项
  useEffect(() => {
    api.project_modelview().then((res: any) => {
      const orgProject = res?.result.data.reduce((acc: any, cur: any) => {
        if (cur.type === 'org') {
          const item = {
            key: cur.id,
            text: cur.name,
          };
          acc.push(item);
        }
        return acc;
      }, []);
      setDropItem(orgProject);
    });
  }, []);

  // 根据节点的变化实时更新 dag_json
  useEffect(() => {
    const temp: any = {};
    elements.forEach(ele => {
      if (isEdge(ele)) {
        const source = elements.filter(el => el.id === ele.source)[0];
        const target = elements.filter(el => el.id === ele.target)[0];

        if (temp[`${target.data.name}`]?.upstream) {
          temp[`${target.data.name}`].upstream.push(`${source.data.name}`);
        } else {
          temp[`${target.data.name}`] = {};
          temp[`${target.data.name}`]['upstream'] = [`${source.data.name}`];
        }
      }
    });

    const dag_json = { ...temp };
    setCurrent({
      ...current,
      ...{ dag_json: JSON.stringify(dag_json, undefined, 4) },
    });
    dispatch(
      updateChanged({
        ...pipelineChanged,
        ...{ dag_json: JSON.stringify(dag_json) },
      }),
    );
  }, [elements]);

  return (
    <div
      style={{
        visibility: settingShow ? 'visible' : 'hidden',
      }}
      className={style.settingContainer}
    >
      <div className={style.settingHeader}>
        <div className={style.headerTitle}>{t('流水线设置')}</div>
        <IconButton
          iconProps={{
            iconName: 'ChromeClose',
            styles: {
              root: {
                fontSize: 12,
                color: '#000',
              },
            },
          }}
          onClick={() => {
            dispatch(toggle());
          }}
        />
      </div>
      <div className={style.settingContent}>
        <div className={style.contentWrapper}>
          <Dropdown
            label={t('项目组')}
            onChange={(e: FormEvent, item?: IDropdownOption) => {
              handleOnChange('project', item || '');
            }}
            selectedKey={selectedItem ? selectedItem.key : undefined}
            placeholder="Select an option"
            options={dropItem}
          />
          <div className={style.splitLine}></div>
          <TextField
            label={t('名称')}
            description={t('英文名(字母、数字、- 组成)，最长50个字符')}
            onChange={(event: FormEvent, value?: string) => {
              handleOnChange('name', value ? value : '');
            }}
            value={current?.name || ''}
            disabled
          />
          <div className={style.splitLine}></div>
          <TextField
            label={t('描述')}
            required
            onChange={(event: FormEvent, value?: string) => {
              handleOnChange('describe', value ? value : '');
            }}
            value={current?.describe || ''}
          />
          {/* <div className={style.splitLine}></div>
          <TextField
            label="命名空间"
            description="部署task所在的命名空间(目前无需填写)"
            value={current?.namespace || ''}
            readOnly
            disabled
          /> */}
          <div className={style.splitLine}></div>
          <Dropdown
            label={t('调度优先级')}
            options={[
              { key: '高优先级', text: '高优先级' },
              { key: '低优先级', text: '低优先级' },
            ]}
          />
          <div className={style.splitLine}></div>
          <Dropdown
            label={t('调度类型')}
            options={[
              { key: 'once', text: 'once' },
              { key: 'crontab', text: 'crontab' },
            ]}
            selectedKey={current?.schedule_type}
            onChange={(event: FormEvent, option?: IDropdownOption) => {
              handleOnChange('schedule_type', `${option?.text}` || '');
            }}
          />
          <div className={style.splitLine}></div>
          <TextField
            label={t('调度周期')}
            description={t('周期任务的时间设定 * * * * * 表示为 minute hour day month week')}
            onChange={(event: FormEvent, value?: string) => {
              handleOnChange('cron_time', value ? value : '');
            }}
            value={current?.cron_time || ''}
          />
          <div className={style.splitLine}></div>

          <div style={{ fontWeight: 600, padding: '5px 0px' }}>{t('监控状态')}</div>
          <Select
            style={{ width: '100%', border: '1px solid rgb(55, 55, 55)' }}
            value={current?.alert_status ? (current?.alert_status || '').split(',') : undefined}
            onChange={(value: any) => {
              handleOnChange('alert_status', (value || []).join(','));
            }}
            mode="multiple"
            options={[
              { label: 'Created', value: 'Created' },
              { label: 'Pending', value: 'Pending' },
              { label: 'Running', value: 'Running' },
              { label: 'Succeeded', value: 'Succeeded' },
              { label: 'Failed', value: 'Failed' },
              { label: 'Unknown', value: 'Unknown' },
              { label: 'Waiting', value: 'Waiting' },
              { label: 'Terminated', value: 'Terminated' },
            ]} />
          {/* <Dropdown
            label="监控状态"
            options={[
              { key: 'Created', text: 'Created' },
              { key: 'Pending', text: 'Pending' },
              { key: 'Running', text: 'Running' },
              { key: 'Succeeded', text: 'Succeeded' },
              { key: 'Failed', text: 'Failed' },
              { key: 'Unknown', text: 'Unknown' },
              { key: 'Waiting', text: 'Waiting' },
              { key: 'Terminated', text: 'Terminated' },
            ]}
            multiSelect
            selectedKey={current?.alert_status}
            onChange={(event: FormEvent, option?: IDropdownOption) => {
              console.log(event, option)
              // handleOnChange('alert_status', `${option?.text}` || '');
            }}
          /> */}
          <div className={style.splitLine}></div>
          <TextField
            label={t('报警人')}
            description={t('每个用户使用英文逗号分隔')}
            onChange={(event: FormEvent, value?: string) => {
              handleOnChange('alert_user', value ? value : '');
            }}
            value={current?.alert_user || ''}
          />
          <div className={style.splitLine}></div>
          {/* <TextField
            label="调度机器"
            description="部署task所在的机器(目前无需填写)"
            onChange={(event: FormEvent, value?: string) => {
              handleOnChange('node_selector', value ? value : '');
            }}
            value={current?.node_selector || ''}
          /> */}
          <div style={{ fontWeight: 600, padding: '5px 0px' }}>{t('补录起点')}</div>
          <DatePicker
            style={{ width: '100%', border: '1px solid rgb(55, 55, 55)' }}
            locale={locale}
            showTime
            value={current?.cronjob_start_time ? moment(current?.cronjob_start_time) : undefined}
            onChange={(date, dateString) => {
              // console.log(date, dateString)
              handleOnChange('cronjob_start_time', dateString);
            }}
            disabledDate={(current) => {
              return current && current > moment().endOf('day');
            }} />

{/*
          <div className={style.splitLine}></div>
          <Dropdown
            label="镜像拉取策略"
            options={[
              { key: 'Always', text: 'Always' },
              { key: 'IfNotPresent', text: 'IfNotPresent' },
            ]}
            selectedKey={current.image_pull_policy}
            onChange={(event: FormEvent, option?: IDropdownOption) => {
              handleOnChange('image_pull_policy', `${option?.text}` || '');
            }}
          /> */}
          <div className={style.splitLine}></div>
          <Dropdown
            label={t('过往依赖')}
            options={[
              { key: 'true', text: t('是'), data: true },
              { key: 'false', text: t('否'), data: false },
            ]}
            selectedKey={`${current.depends_on_past}`}
            onChange={(event: FormEvent, option?: IDropdownOption) => {
              handleOnChange('depends_on_past', option?.data);
            }}
          />
          <div className={style.splitLine}></div>
          <TextField
            label={t('最大激活运行数')}
            description={t('当前pipeline可同时运行的任务流实例数目')}
            onChange={(event: FormEvent, value?: string) => {
              handleOnChange('max_active_runs', value ? +value : '');
            }}
            value={current?.max_active_runs || ''}
            required
          />
          <div className={style.splitLine}></div>
          <TextField
            label={t('任务并行数')}
            description={t('pipeline中可同时运行的task数目')}
            onChange={(event: FormEvent, value?: string) => {
              handleOnChange('parallelism', value ? +value : '');
            }}
            value={current?.parallelism || ''}
            required
          />
          <div className={style.splitLine}></div>
          <TextField
            label={t('流向图')}
            onChange={(event: FormEvent, value?: string) => {
              handleOnChange('dag_json', value ? value : '{}');
            }}
            value={current?.dag_json || '{}'}
            multiline
            autoAdjustHeight
            disabled
          />
          <div className={style.splitLine}></div>
          <TextField
            label={t('全局环境变量')}
            description={t('为每个task都添加的公共参数')}
            onChange={(event: FormEvent, value?: string) => {
              handleOnChange('global_env', value ? value : '');
            }}
            multiline
            autoAdjustHeight
            value={current?.global_env || ''}
          />
        </div>
      </div>
    </div>
  );
};

export default Setting;
