import React, { useState, useEffect } from 'react';
import api from '@src/api';
import { CopyToClipboard } from 'react-copy-to-clipboard';
import RecallFrameworkTable from './componentsChilder/RecallFrameworkTable';
import PolicyInformation from './componentsChilder/PolicyInformation';
import RegisterRecall from './componentsChilder/RegisterRecall';
import RegisterAuxiliary from './componentsChilder/RegisterAuxiliary';
import ModifyInformation from './componentsChilder/ModifyInformation';
import RegisterAuxiliaryTapy from './componentsChilder/RegisterAuxiliaryTapy';
import InformationDisplay from './componentsChilder/InformationDisplay';
import ConfigurationValidation from './componentsChilder/ConfigurationValidation';
import { SolutionOutlined, CopyOutlined } from '@ant-design/icons';

import { Row, Col, Button, Input, Select, message, Tooltip, Tag } from 'antd';
import './RecallFramework.css';
const style = { padding: '8px 0', display: 'flex', alignItems: 'center' };

function statusValue(value: any) {
  switch (value) {
    case 0:
      return '注册成功';
    case 1:
      return '测试发布';
    case 2:
      return '正式发布';
    case -1:
      return '失效';
    case -2:
      return '删除';
    default:
      return '未知';
  }
}
function appValue(value: any) {
  switch (value) {
    case 1:
      return 'k歌';
    case 2:
      return 'k歌国际版';
    case 4:
      return 'QQ音乐';
    case 8:
      return '音兔';
    case 16:
      return '酷狗音乐';
    case 32:
      return '酷我音乐';
    case 64:
      return '爱听卓乐';
    case 128:
      return '懒人畅听';
    case 256:
      return '其它';
    default:
      return '未知';
  }
}

function RecallFramework() {
  const columns = [
    {
      title: '策略ID',
      dataIndex: 'strategy_id',
      align: 'center',
      width: 80,
      /* eslint-disable */
      render: (text: any, record: any) => {
        return (
          <div>
            <div style={{ whiteSpace: 'normal' }}>
              {record.strategy_id}
              <CopyToClipboard text={record.strategy_id} onCopy={() => message.success('复制成功~')}>
                <CopyOutlined style={{ paddingLeft: 2 }} />
              </CopyToClipboard>
            </div>
          </div>
        );
      },
    },
    {
      title: '应用',
      dataIndex: 'app',
      align: 'center',
      width: 104,
      /* eslint-disable */
      render: (text: any, record: any) => {
        return <div>{appValue(record.app)}</div>;
      },
    },
    {
      title: '召回组件',
      dataIndex: 'recall_mark',
      align: 'center',
      // width: 200,
      ellipsis: true,
      /* eslint-disable */
      render: (text: any, record: any) => {
        return (
          <div>
            <div style={{ whiteSpace: 'normal' }}>
              {record.recall_mark}
              <CopyToClipboard text={record.recall_mark} onCopy={() => message.success('复制成功~')}>
                <CopyOutlined style={{ paddingLeft: 2 }} />
              </CopyToClipboard>
            </div>
          </div>
        );
      },
    },
    {
      title: '中间件',
      dataIndex: 'middleware_marks',
      align: 'center',
      // width: 220,
      ellipsis: true,
      /* eslint-disable */
      render: (text: any, record: any) => {
        return (
          <div>
            <div style={{ whiteSpace: 'normal' }}>
              {record.middleware_marks}
              <CopyToClipboard text={record.middleware_marks} onCopy={() => message.success('复制成功~')}>
                <CopyOutlined style={{ paddingLeft: 2 }} />
              </CopyToClipboard>
            </div>
          </div>
        );
      },
    },
    {
      title: '过滤组件',
      dataIndex: 'filter_marks',
      align: 'center',
      // width: 220,
      ellipsis: true,
      /* eslint-disable */
      render: (text: any, record: any) => {
        return (
          <div>
            <div style={{ whiteSpace: 'normal' }}>
              {record.filter_marks}
              <CopyToClipboard text={record.filter_marks} onCopy={() => message.success('复制成功~')}>
                <CopyOutlined style={{ paddingLeft: 2 }} />
              </CopyToClipboard>
            </div>
          </div>
        );
      },
    },
    {
      title: '排序组件',
      dataIndex: 'sort_marks',
      align: 'center',
      // width: 220,
      ellipsis: true,
      /* eslint-disable */
      render: (text: any, record: any) => {
        return (
          <div>
            <div style={{ whiteSpace: 'normal' }}>
              {record.sort_marks}
              <CopyToClipboard text={record.sort_marks} onCopy={() => message.success('复制成功~')}>
                <CopyOutlined style={{ paddingLeft: 2 }} />
              </CopyToClipboard>
            </div>
          </div>
        );
      },
    },
    {
      title: '负责人',
      dataIndex: 'admin',
      align: 'center',
      width: 100,
    },
    {
      title: '召回配置',
      dataIndex: 'recall_config',
      align: 'center',
      // width: 220,
      ellipsis: true,
      /* eslint-disable */
      render: (text: any, record: any) => {
        return (
          <div>
            <Tooltip placement="topLeft" title={record.recall_config}>
              {record.recall_config}
            </Tooltip>

            <br />
            <CopyToClipboard text={record.recall_config} onCopy={() => message.success('复制成功~')}>
              <CopyOutlined style={{ paddingLeft: 2 }} />
            </CopyToClipboard>
          </div>
        );
      },
    },
    {
      title: '状态',
      dataIndex: 'status',
      align: 'center',
      width: 82,
      ellipsis: true,
      /* eslint-disable */
      render: (text: any, record: any) => {
        let color = '';
        if (record.status === 0) {
          color = '#f50';
        } else if (record.status === 1) {
          color = '#87d068';
        } else if (record.status === 2) {
          color = '#108ee9';
        } else if (record.status === -1) {
          color = 'orange';
        } else if (record.status === -2) {
          color = '#cd201f';
        }
        return <Tag color={color}>{statusValue(record.status)}</Tag>;
      },
    },
    {
      title: '概括描述',
      dataIndex: 'brief_des',
      align: 'center',
    },
    {
      title: '描述',
      dataIndex: 'des',
      align: 'center',
    },
    {
      title: '操作',
      dataIndex: 'action',
      align: 'center',

      /* eslint-disable */
      render: (text: any, record: any) => (
        <div className="ButtonClass">
          {/* <a>Invite {record.name}</a> */}
          <Button type="dashed" onClick={() => showModalFuncModify({ title: '修改策略信息', record })}>
            修改
          </Button>
          <Button
            danger
            disabled={record.status === 2 || record.status === 1 ? false : true}
            onClick={() => rollbackFuncModify(record)}
          >
            回滚
          </Button>
          <Button disabled={record.status === 0 ? false : true} onClick={() => test_releaseFuncModify(record)}>
            测试发布
          </Button>
          <Button
            disabled={record.status === 1 ? false : true}
            type="primary"
            onClick={() => real_releaseFuncModify(record)}
          >
            正式发布
          </Button>
          {/* <Divider type="vertical" />
          <a>Delete</a> */}
        </div>
      ),
      /* eslint-disable */
    },
  ];

  const [columnsValue, setcolumnsValue] = useState(columns),
    [dataValue, setDataValue] = useState([]),
    [visiblePolicyInformation, setVisiblePolicyInformation] = useState(false),
    [visibleRegisterRecall, setvisibleRegisterRecall] = useState(false),
    [visibleRegisterAuxiliary, setvisibleRegisterAuxiliary] = useState(false),
    [visibleRegisterAuxiliaryTapy, setvisibleRegisterAuxiliaryTapy] = useState(false),
    [ModifyInformationvisible, setModifyInformationvisible] = useState(false),
    [visibleInformationDisplay, setvisibleInformationDisplay] = useState(false),
    [visibleConfigurationValidation, setvisibleConfigurationValidation] = useState(false),
    [valueListTable, setvalueListTable] = useState({
      title: '编辑策略信息',
      record: { strategy_id: undefined, version: undefined, status: undefined },
    }),
    [fromRow, setFromRow] = useState({
      strategy_ids: [],
      component_marks: [],
      admins: [],
    }),
    [fromRowSelect, setFromRowSelect] = useState({
      status: [],
      apps: [],
    }),
    [data1, setData] = useState([]),
    [data2, setData2] = useState([]),
    [data3, setData3] = useState([]),
    [data4, setData4] = useState([]),
    // [data5, setData5] = useState({
    //   address: '',
    //   host: '',
    //   db: '',
    //   collection: '',
    //   auth: '',
    // }),
    [data6, setData6] = useState({});

  const statusOPTIONS = [
    { key: 0, value: '注册成功' },
    { key: 1, value: '测试发布' },
    { key: 2, value: '正式发布' },
    { key: -1, value: '失效' },
    { key: -2, value: '删除' },
  ];
  const appsOPTIONS = [
    { key: 1, value: 'k歌' },
    { key: 2, value: 'k歌国际版' },
    { key: 4, value: 'QQ音乐' },
    { key: 8, value: '音兔' },
    { key: 16, value: '酷狗音乐' },
    { key: 32, value: '酷我音乐' },
    { key: 64, value: '爱听卓乐' },
    { key: 128, value: '懒人畅听' },
    { key: 256, value: '其它' },
  ];
  // componentDidMount
  useEffect(() => {
    let queryFuncValueFirst = Object.assign(fromRow, fromRowSelect);
    queryFuncTable(queryFuncValueFirst);
  }, []);
  //封装高阶函数收集input数据
  const fromFunChanger = (value: string) => {
    return (e: any) => {
      setFromRow({
        ...fromRow,
        [value]: e.target.value.split(';').splice(''),
      });
    };
  };
  const fromFunChangerTwo = (value: any) => {
    return (e: any) => {
      console.log(e, '66');
      setFromRowSelect({
        ...fromRowSelect,
        [value]: e,
      });
    };
  };

  //查询
  const queryFunc = () => {
    function strategy_idsValue(strategy_ids: any, stringValue: string) {
      if (strategy_ids[0] === '') {
        return (strategy_ids = []);
      } else {
        if (stringValue === 'component_marks') {
          return (strategy_ids = strategy_ids);
        }
        if (stringValue === 'admins') {
          return (strategy_ids = strategy_ids);
        }
        if (stringValue === 'strategy_ids') {
          return strategy_ids.map(Number);
        }
      }
    }
    let queryFuncValue = Object.assign(
      {
        ...fromRow,
        component_marks: strategy_idsValue(fromRow.component_marks, 'component_marks'),
        strategy_ids: strategy_idsValue(fromRow.strategy_ids, 'strategy_ids'),
        admins: strategy_idsValue(fromRow.admins, 'admins'),
      },
      fromRowSelect,
    );

    queryFuncTable(queryFuncValue);
  };
  function queryFuncTable(queryFuncValue: any) {
    api.get_strategys(queryFuncValue).then((item: any) => {
      if (item.retcode === 0) {
        message.success('查询成功');
      } else if (item.retmsg) {
        message.error(`失败原因: ${item.retmsg}`);
      }
      setDataValue(item.result.data);
    });
  }

  //表格中的修改
  const showModalFuncModify = (valueList: any) => {
    if (valueList && valueList.record !== undefined && valueList.title === '修改策略信息') {
      if (valueList.record.middleware_marks.includes(';')) {
        valueList.record.middleware_marks = valueList.record.middleware_marks.split(';');
      } else if (!valueList.record.middleware_marks) {
        valueList.record.middleware_marks = [];
      }
      if (valueList.record.filter_marks.includes(';')) {
        valueList.record.filter_marks = valueList.record.filter_marks.split(';');
      } else if (!valueList.record.filter_marks) {
        valueList.record.filter_marks = [];
      }
      if (valueList.record.sort_marks.includes(';')) {
        valueList.record.sort_marks = valueList.record.sort_marks.split(';');
      } else if (!valueList.record.sort_marks) {
        valueList.record.sort_marks = [];
      }
    }
    setvalueListTable(valueList);
    setVisiblePolicyInformation(true);

    api.get_components_mark(['filter', 'sort', 'recall', 'middleware']).then(d => {
      const { result } = d;
      let Sunarraym: any = [];
      let filterValue: any = [];
      let middlewareValue: any = [];
      let sortValue: any = [];
      if (result) {
        // const { filter, middleware, recall, sort } = result.data;
        const { recall, middleware, filter, sort } = result.data;
        Sunarraym = [].concat(recall);
        filterValue = [].concat(filter);
        middlewareValue = [].concat(middleware);
        sortValue = [].concat(sort);

        setData(Sunarraym);
        setData2(filterValue);
        setData3(middlewareValue);
        setData4(sortValue);
      }
    });
  };

  const rollbackFuncModify = (valueList: any) => {
    api.rollback(valueList.strategy_id).then(item => {
      if (item.retcode === 0) {
        message.success('回滚成功');
        queryFuncTable(Object.assign(fromRow, fromRowSelect));
      } else if (item.retmsg) {
        message.error(`失败原因: ${item.retmsg}`);
      }
    });
  };
  const test_releaseFuncModify = (valueList: any) => {
    api.test_release(valueList.strategy_id).then(item => {
      if (item.retcode === 0) {
        message.success('测试发布策略成功');
        queryFuncTable(Object.assign(fromRow, fromRowSelect));
      } else if (item.retmsg) {
        message.error(`失败原因: ${item.retmsg}`);
      }
    });
  };
  const real_releaseFuncModify = (valueList: any) => {
    api.real_release(valueList.strategy_id).then(item => {
      if (item.retcode === 0) {
        message.success('正式发布策略成功');
        queryFuncTable(Object.assign(fromRow, fromRowSelect));
      } else if (item.retmsg) {
        message.error(`失败原因: ${item.retmsg}`);
      }
    });
  };

  //添加召回策略
  const showModalFunc = () => {
    setVisiblePolicyInformation(true);

    api.get_components_mark(['filter', 'sort', 'recall', 'middleware']).then(d => {
      const { result } = d;
      let Sunarraym: any = [];
      let filterValue: any = [];
      let middlewareValue: any = [];
      let sortValue: any = [];
      if (result) {
        const { recall, middleware, filter, sort } = result.data;

        Sunarraym = [].concat(recall);
        filterValue = [].concat(filter);
        middlewareValue = [].concat(middleware);
        sortValue = [].concat(sort);

        setData(Sunarraym);
        setData2(filterValue);
        setData3(middlewareValue);
        setData4(sortValue);
      }
    });

    showModalFuncModify({ title: '编辑策略信息', record: {} });
    console.log('添加召回策略');
  };
  const onCreate = (values: any) => {
    if (valueListTable && valueListTable.title === '编辑策略信息') {
      const valueSun = {
        ...values,
        filter_marks: Array.isArray(values.filter_marks) ? values.filter_marks.join(';') : values.filter_marks,
        middleware_marks: Array.isArray(values.middleware_marks)
          ? values.middleware_marks.join(';')
          : values.middleware_marks,
        sort_marks: Array.isArray(values.sort_marks) ? values.sort_marks.join(';') : values.sort_marks,
        // recall_config: JSON.stringify(data5),
        recall_config:
          typeof values.recall_config === 'string' ? values.recall_config : JSON.stringify(values.recall_config),
      };
      api.add_strategy(valueSun).then(item => {
        if (item.retcode === 0) {
          message.success('添加召回策略成功');
          setVisiblePolicyInformation(false);
          queryFuncTable(Object.assign(fromRow, fromRowSelect));
        } else if (item.retmsg) {
          message.error(`失败原因: ${item.retmsg}`);
        }
      });
    } else {
      let valueModif = {
        ...values,
        filter_marks: Array.isArray(values.filter_marks) ? values.filter_marks.join(';') : values.filter_marks,
        middleware_marks: Array.isArray(values.middleware_marks)
          ? values.middleware_marks.join(';')
          : values.middleware_marks,
        sort_marks: Array.isArray(values.sort_marks) ? values.sort_marks.join(';') : values.sort_marks,
        strategy_id: valueListTable.record.strategy_id,
        version: valueListTable.record.version,
        status: valueListTable.record.status,
        recall_config:
          typeof values.recall_config === 'string' ? values.recall_config : JSON.stringify(values.recall_config),
      };
      api.modify_strategy(valueModif).then(item => {
        if (item.retcode === 0) {
          message.success('修改召回策略成功');
          setVisiblePolicyInformation(false);
          queryFuncTable(Object.assign(fromRow, fromRowSelect));
        } else if (item.retmsg) {
          message.error(`失败原因: ${item.retmsg}`);
        }
      });
    }
  };

  //注册召回组件
  const RegisterRecallFunc = () => {
    setvisibleRegisterRecall(true);
  };
  const onCreateRegisterRecall = (values: any) => {
    api.register_recall_component(values).then(item => {
      if (item.retcode === 0) {
        message.success('注册召回组件成功');
        setvisibleRegisterRecall(false);
      } else if (item.retmsg) {
        message.error(`失败原因: ${item.retmsg}`);
      }
    });
  };
  //注册辅助组件
  const RegisterRegisterAuxiliaryFunc = () => {
    setvisibleRegisterAuxiliary(true);
  };
  const onCreateRegisterAuxiliary = (values: any) => {
    let register_recall_componentSum = {
      ...values,
      config: JSON.stringify(data6),
    };
    api.register_assistant_component(register_recall_componentSum).then(item => {
      if (item.retcode === 0) {
        message.success('注册辅助组件成功');
        setvisibleRegisterAuxiliary(false);
      } else if (item.retmsg) {
        message.error(`失败原因: ${item.retmsg}`);
      }
    });
    setData6({});
  };
  //注册辅助组件类型
  const RegisterAuxiliaryTapyFunc = () => {
    setvisibleRegisterAuxiliaryTapy(true);
  };
  const onCreateRegisterAuxiliaryTapy = (values: any) => {
    api.register_assistant_component_type(values).then(item => {
      if (item.retcode === 0) {
        message.success('注册辅助组件类型成功');
        setvisibleRegisterAuxiliaryTapy(false);
      } else if (item.retmsg) {
        message.error(`失败原因: ${item.retmsg}`);
      }
    });
  };

  //配置效验
  const ConfigurationValidationFunc = () => {
    setvisibleConfigurationValidation(true);
  };
  //查看组件信息
  const InformationDisplayFunc = () => {
    setvisibleInformationDisplay(true);
  };

  //修改组件信息
  const onCreateModifyInformationFunc = () => {
    setModifyInformationvisible(true);
  };
  const onCreateModifyInformation = (values: any) => {
    api.mod_component_info(values).then(item => {
      if (item.retcode === 0) {
        message.success('修改组件信息成功');
        setModifyInformationvisible(false);
      } else if (item.retmsg) {
        message.error(`失败原因: ${item.retmsg}`);
      }
    });
  };
  //
  const aLickTo = () => {
    window.open('https://docs.qq.com/sheet/DZGZuTlJxTHRycWFz?tab=BB08J2');
  };
  // const getAddEdit = (val: any) => {
  //   console.log(val, '-----==========99999');

  //   setData5(val);
  // };
  const getAddEdit2 = (val: any) => {
    setData6(val);
  };
  return (
    <>
      <div className="RecallFrameworkFirst">
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <h1>统一物料召回</h1>
          <div className="zhanhuiName" style={{ fontSize: '20px', marginRight: '5%' }} onClick={aLickTo}>
            <SolutionOutlined />
            <span>吐槽</span>
          </div>
        </div>

        <hr />

        <Row gutter={16}>
          <Col className="gutter-row" span={4}>
            <div style={style}>
              <span>策略ID：</span>
              <Input
                placeholder="请输入策略ID"
                value={fromRow.strategy_ids}
                onChange={fromFunChanger('strategy_ids')}
              />
            </div>
          </Col>
          <Col className="gutter-row" span={4}>
            <div style={style}>
              <span style={{ width: '33%' }}>组件mark：</span>
              <Input
                placeholder="请输入组件mark"
                value={fromRow.component_marks}
                onChange={fromFunChanger('component_marks')}
              />
            </div>
          </Col>
          <Col className="gutter-row" span={4}>
            <div style={style}>
              <span>应用：</span>
              <Select
                mode="multiple"
                placeholder="Inserted are removed"
                value={fromRowSelect.apps}
                onChange={fromFunChangerTwo('apps')}
                style={{ width: '200px' }}
              >
                {appsOPTIONS.map(item => {
                  return (
                    <Select.Option key={item.key} value={item.key}>
                      {item.value}
                    </Select.Option>
                  );
                })}
              </Select>
            </div>
          </Col>
          <Col className="gutter-row" span={4}>
            <div style={style}>
              <span>状态：</span>
              <Select
                mode="multiple"
                placeholder="Inserted are removed"
                value={fromRowSelect.status}
                onChange={fromFunChangerTwo('status')}
                style={{ width: '200px' }}
              >
                {statusOPTIONS.map(item => {
                  return (
                    <Select.Option key={item.key} value={item.key}>
                      {item.value}
                    </Select.Option>
                  );
                })}
              </Select>
            </div>
          </Col>
          <Col className="gutter-row" span={4}>
            <div style={style}>
              <span>负责人：</span>
              <Input placeholder="请输入负责人" value={fromRow.admins} onChange={fromFunChanger('admins')} />
            </div>
          </Col>
          <Col className="gutter-row2" span={4}>
            <Button type="primary" style={{ marginTop: '8px' }} onClick={queryFunc}>
              查询
            </Button>
          </Col>
        </Row>
        <div className="Header_end">
          <Button type="primary" onClick={showModalFunc}>
            添加召回策略
          </Button>
          <Button type="primary" onClick={RegisterRecallFunc}>
            注册召回组件
          </Button>
          <Button type="primary" onClick={RegisterRegisterAuxiliaryFunc}>
            注册辅助组件
          </Button>
          <Button type="primary" onClick={RegisterAuxiliaryTapyFunc}>
            注册辅助组件类型
          </Button>
          <Button type="primary" onClick={ConfigurationValidationFunc}>
            配置效验
          </Button>
          <Button type="primary" onClick={InformationDisplayFunc}>
            查看组件信息
          </Button>
          <Button type="primary" onClick={onCreateModifyInformationFunc}>
            修改组件信息
          </Button>
        </div>
        <hr style={{ margin: '20px 0' }} />
        <RecallFrameworkTable columns={columnsValue} data={dataValue} />
        <hr />
        {visiblePolicyInformation ? (
          <PolicyInformation
            visible={visiblePolicyInformation}
            onCreate={onCreate}
            valueListTable={valueListTable}
            valueData={data1}
            valueData2={data2}
            valueData3={data3}
            valueData4={data4}
            // getAddEdit={getAddEdit}
            onCancel={() => {
              setVisiblePolicyInformation(false);
            }}
          />
        ) : null}
        <RegisterRecall
          visible={visibleRegisterRecall}
          onCreate={onCreateRegisterRecall}
          onCancel={() => {
            setvisibleRegisterRecall(false);
          }}
        />
        <RegisterAuxiliary
          getAddEdit2={getAddEdit2}
          visible={visibleRegisterAuxiliary}
          onCreate={onCreateRegisterAuxiliary}
          onCancel={() => {
            setvisibleRegisterAuxiliary(false);
          }}
        />
        <RegisterAuxiliaryTapy
          visible={visibleRegisterAuxiliaryTapy}
          onCreate={onCreateRegisterAuxiliaryTapy}
          onCancel={() => {
            setvisibleRegisterAuxiliaryTapy(false);
          }}
        />
        <ConfigurationValidation
          visible={visibleConfigurationValidation}
          onCancel={() => {
            setvisibleConfigurationValidation(false);
          }}
        />
        <InformationDisplay
          visible={visibleInformationDisplay}
          onCancel={() => {
            setvisibleInformationDisplay(false);
          }}
        />

        <ModifyInformation
          visible={ModifyInformationvisible}
          onCreate={onCreateModifyInformation}
          onCancel={() => {
            setModifyInformationvisible(false);
          }}
        />
      </div>
    </>
  );
}
export default RecallFramework;
