import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Button, Input, Select, Radio, message, Form } from 'antd';
import { PlusOutlined, MenuOutlined } from '@ant-design/icons';
import { useHistory } from 'react-router-dom';
import api from '@src/api';
// import { SortableHandle } from 'react-sortable-hoc';
import FeatureConfigurationTable from '../SceneModeTable/FeatureConfigurationTable';
import FeatureConfigurationTableSort from '../SceneModeTable/FeatureConfigurationTableSort';
import CollectionCreateForm from './model';
import ModelOne from './modelOne';
import './FeatureConfiguration.css';

const style = { padding: '8px 0', display: 'flex', alignItems: 'center' };
// const DragHandle = SortableHandle(() => <MenuOutlined style={{ cursor: 'grab', color: '#999' }} />);

const comStore = {
  uuid: 0,
  getUUid() {
    return this.uuid++;
  },
};

export default function FeatureConfiguration() {
  const history = useHistory();
  const [form] = Form.useForm();
  const [visible, setVisible] = useState(false);
  const [Input_item, setInput_item] = useState({ input_item: [{ cre_time: '' }] });
  const [Op_item, setOp_item] = useState({ opItems: [{ cre_time: '' }], op_item: [{ cre_time: '' }] });
  const [hashIdValue, sethashIdValue] = useState('');
  const [visibleOne, setVisibleOne] = useState(false);

  const [featureDisplayOpValue, setfeatureDisplayOpValue] = useState([{ op_name: '', op_id: '' }]);
  const [fe_proc_config, setfe_proc_config] = useState({ fe_proc_id: '', cre_time: '', last_mod: '', model_id: '' });

  const [HASHFUNCValue, setHASHFUNCValue] = useState([{ label: '', value: '' }]);

  const [CONTEXTTYPEValue, setCONTEXTTYPEValue] = useState([{ label: '', value: '' }]);
  const [FEATURETYPE, setFEATURETYPE] = useState([{ label: '', value: '' }]);
  const [feature_typeShow, setfeature_typeShow] = useState({ value: '' }),
    [valueListTable, setvalueListTable] = useState({
      title: '编辑算子处理配置',
      record: { op_item_id: '', strategy_id: undefined, version: undefined, status: undefined },
    }),
    [valueListTableOne, setvalueListTableOne] = useState({
      title: '编辑输入特征配置',
      record: {
        strategy_id: undefined,
        version: undefined,
        status: undefined,
        input_item_id: undefined,
        ctype: '',
        dtype: '',
        input_desc: '',
        input_seq: '',
        raw_fe_name: '',
      },
    }),
    [fromRow, setFromRow] = useState({
      fe_proc_name: '',
      fe_proc_desc: '',
      owner_rtxs: '',
      need_hash: 1,
      slot_flat: 1,
      fe_proc_id: { fe_proc_id: '' },
    }),
    [fromRowSelect, setFromRowSelect] = useState({
      hash_func: [],
    }),
    [scene_id_ID, setscene_id_ID] = useState({ scene_id: '', title: '' }),
    [isomerismFeatureInfoValue, setisomerismFeatureInfoValue] = useState([
      { hashId: '', feature_name: '', input_desc: '', raw_fe_name: '', feature_type: '' },
    ]),
    [Op_op_id, setOp_op_id] = useState(undefined),
    [Datalist, setDatalist] = useState([]),
    [dataList2, setDataList2] = useState([]);
  const dtypeValue = (dtypeKey: string) => {
    switch (dtypeKey) {
      case '6':
        return 'emb_ints';
      case '4':
        return 'numer_int';
      case '3':
        return 'emb_floats';
      case '1':
        return 'number_float';
      default:
        return null;
    }
  };
  const ctypeValue = (ctypeValueKey: string) => {
    switch (ctypeValueKey) {
      case '3':
        return 'context';
      case '2':
        return 'item';
      case '1':
        return 'user';
      default:
        return null;
    }
  };
  const out_fe_typeVlaue = (out_fe_typeKey: string) => {
    switch (out_fe_typeKey) {
      case '2':
        return 'sparse';
      case '1':
        return 'dense';
      default:
        return null;
    }
  };
  const columns = [
    // {
    //   dataIndex: 'id',
    //   width: 60,
    //   className: 'drag-visible',
    //   render: (dom: any, rowData: any, index: any) => {
    //     return (
    //       <span style={{ display: 'flex' }}>
    //         <DragHandle />
    //         {`[${index}]`}
    //       </span>
    //     );
    //   },
    // },
    {
      title: 'id',
      dataIndex: 'input_seq',
      width: 60,
      render: (dom: any, rowData: any, index: any) => {
        return <span>{index + 1}</span>;
      },
    },
    {
      title: '特征slot',
      dataIndex: 'raw_fe_name',
      render: (_: any, row: any) => {
        return row.raw_fe_name;
      },
    },
    {
      title: '哈希id',
      dataIndex: 'hashid',
      render: (_: any, row: any) => {
        return row.hashid;
      },
    },
    {
      title: '特征名字',
      dataIndex: 'input_desc',
      render: (_: any, row: any) => {
        return row.input_desc;
      },
    },
    {
      title: '特征类型',
      dataIndex: 'dtype',
      render: (_: any, row: any) => {
        return dtypeValue(row.dtype);
      },
    },
    {
      title: '上下文类型',
      dataIndex: 'ctype',
      render: (_: any, row: any) => {
        return ctypeValue(row.ctype);
      },
    },
    {
      title: '创建时间',
      dataIndex: 'cre_time',
      width: 140,
    },
    {
      title: '上次修改时间',
      dataIndex: 'last_mod',
      width: 140,
    },
    {
      title: '操作',
      dataIndex: 'address666',
      /* eslint-disable */
      render: (text: any, record: any) => (
        <div className="ButtonClass">
          <Button type="dashed" onClick={() => showModalFuncModifyOne({ title: '修改特征配置', record })}>
            修改
          </Button>
          <Button danger onClick={() => removeOne(record)}>
            删除
          </Button>
        </div>
      ),
      /* eslint-disable */
    },
  ];

  const columnsSort = [
    // {
    //   title: 'id',
    //   dataIndex: 'id',

    //   render: (dom: any, rowData: any, index: any) => {
    //     return (
    //       <span style={{ display: 'flex' }}>
    //         <DragHandle />
    //         {`[${index}]`}
    //       </span>
    //     );
    //   },
    // },
    {
      title: '序号',
      dataIndex: 'index',
      render: (dom: any, rowData: any, index: any) => {
        return <span>{`${index}`}</span>;
      },
    },
    {
      title: '算子名字',
      dataIndex: 'op_item_name',
    },
    {
      title: '参数1',
      dataIndex: 'age',
      render: (_: any, row: any) => {
        return row.params ? row.params.split(',').slice(0, 1) : '';
      },
    },
    {
      title: '参数2',
      dataIndex: 'name',

      render: (_: any, row: any) => {
        return row.params ? row.params.split(',').slice(1, 2) : '';
      },
    },

    {
      title: '参数3',
      render: (_: any, row: any) => {
        return row.params ? row.params.split(',').slice(2, 3) : '';
      },
    },
    {
      title: '输出特征slot',
      dataIndex: 'out_index',
    },
    {
      title: '输出类型',
      dataIndex: 'out_fe_type',
      render: (_: any, row: any) => {
        return out_fe_typeVlaue(row.out_fe_type);
      },
    },
    {
      title: '创建时间',
      dataIndex: 'cre_time',
      width: 140,
    },
    {
      title: '上次修改时间',
      dataIndex: 'last_mod',
      width: 140,
    },
    {
      title: '操作',
      dataIndex: 'address5',

      /* eslint-disable */
      render: (text: any, record: any) => (
        <div className="ButtonClass">
          {/* <a>Invite {record.name}</a> */}
          <Button type="dashed" onClick={() => showModalFuncModify({ title: '修改算子处理配置', record })}>
            修改
          </Button>
          <Button danger onClick={() => remove(record)}>
            删除
          </Button>
        </div>
      ),
      /* eslint-disable */
    },
  ];

  const removeOne = (record: any) => {
    const input_item = { inputItems: [{ ...record, isDel: 1 }] };

    api.featureUpdateInputItemPost(input_item).then(item => {
      if (item.retcode === 0) {
        setDatalist(Datalist.filter((item: any) => item.input_item_id !== record.input_item_id));
        message.success('删除成功');
      } else if (item.retmsg) {
        message.error(`失败原因: ${item.retmsg}`);
      }
    });
  };
  const remove = (record: any) => {
    const op_item = { opItems: [{ ...record, isDel: 1 }] };

    api.featureUpdateOpItemPost(op_item).then(item => {
      if (item.retcode === 0) {
        setDataList2(
          dataList2.filter((item: any) => {
            return item.op_item_id !== record.op_item_id;
          }),
        );
        message.success('删除成功');
      } else if (item.retmsg) {
        message.error(`失败原因: ${item.retmsg}`);
      }
    });
  };
  //  one table
  const onCreateOne = (values: any) => {
    if (valueListTableOne && valueListTableOne.title === '编辑输入特征配置') {
      const newData: any = [...Datalist];
      // const newData: any = [];

      newData.push({ ...values, dtype: feature_typeShow.value, hashid: hashIdValue, input_seq: comStore.getUUid() });

      let input_item: any = { input_item: [...newData] };

      setInput_item(input_item);
      setDatalist(newData);
      setVisibleOne(false);
    } else if (valueListTableOne && valueListTableOne.title === '修改特征配置') {
      let newData: any = [];
      valueListTableOne.record.ctype = values.ctype;
      valueListTableOne.record.dtype = values.dtype;
      valueListTableOne.record.input_desc = values.input_desc;
      valueListTableOne.record.raw_fe_name = values.raw_fe_name;

      newData.push({
        ...values,
        input_item_id: valueListTableOne.record.input_item_id,
        input_seq: valueListTableOne.record.input_seq,
      });
      let input_item: any = { inputItems: [...newData] };

      api.featureUpdateInputItemPost(input_item).then(item => {
        if (item.retcode === 0) {
          fe_proc_idHandle(scene_id_ID);
        } else if (item.retmsg) {
          message.error(`失败原因: ${item.retmsg}`);
        }
      });
      setInput_item(input_item);
      setDatalist(newData);
      setVisibleOne(false);
    }
  };
  const onCreate = (values: any) => {
    if (valueListTable && valueListTable.title === '编辑算子处理配置') {
      let newData2: any = [...dataList2];
      newData2.push({
        ...values,
        op_seq: comStore.getUUid(),
        out_index: Number(values.out_index),
        op_id: Op_op_id,
        params: values.parameter1 + ',' + values.parameter2 + ',' + values.parameter3,
      });
      let op_item: any = { op_item: [...newData2] };
      setOp_item(op_item);
      setDataList2(newData2);
      setVisible(false);
      setvalueListTable({
        title: '编辑算子处理配置',
        record: { op_item_id: '', strategy_id: undefined, version: undefined, status: undefined },
      });
    } else if (valueListTable && valueListTable.title === '修改算子处理配置') {
      let newData2: any = [...dataList2];

      let op_item: any = {
        opItems: [
          {
            ...values,
            out_index: Number(values.out_index),
            op_item_id: valueListTable.record.op_item_id,
            op_id: Op_op_id,
            params: values.parameter1 + ',' + values.parameter2 + ',' + values.parameter3,
          },
        ],
      };

      api.featureUpdateOpItemPost(op_item).then(item => {
        if (item.retcode === 0) {
          fe_proc_idHandle(scene_id_ID);
        } else if (item.retmsg) {
          message.error(`失败原因: ${item.retmsg}`);
        }
      });

      setOp_item(op_item);
      setDataList2(newData2);
      setVisible(false);
    }
  };
  //  表格中的修改
  const showModalFuncModify = (valueList: any) => {
    const _valueListObject = JSON.parse(JSON.stringify(valueList));
    setvalueListTable(_valueListObject);
    setVisible(true);
  };
  const showModalFuncTow = () => {
    setvalueListTable({
      title: '编辑算子处理配置',
      record: { op_item_id: '', strategy_id: undefined, version: undefined, status: undefined },
    });
    setVisible(true);
  };

  function fromFunChanger(value: any) {
    return (e: any) => {
      setFromRow({
        ...fromRow,
        [value]: e.target.value,
      });
    };
  }

  const fromFunChangerTwo = (value: any) => {
    return (e: any) => {
      setFromRowSelect({
        ...fromRowSelect,
        [value]: e,
      });
    };
  };

  useEffect(() => {
    // 获取算子
    api.featureDisplayOpGetQuest().then(item => {
      if (item.retcode === 0) {
        setfeatureDisplayOpValue(item.result);
      } else if (item.retmsg) {
        message.error(`失败原因: ${item.retmsg}`);
      }
    });

    function isomerismFeatureInfoValueHandel() {
      let p = new Promise(function (resolve, reject) {
        let setFromRow_value: any = {};

        if (history.location.state) {
          //  判断当前有参数
          setFromRow_value = history.location.state;

          sessionStorage.setItem('featureConfigurationKeyID', JSON.stringify(setFromRow_value)); // 存入到sessionStorage中
          setscene_id_ID(setFromRow_value);
          setFromRow(setFromRow_value);
          setFromRowSelect(setFromRow_value);
        } else {
          setFromRow_value = JSON.parse(sessionStorage.getItem('featureConfigurationKeyID') || ''); // 当state没有参数时，取sessionStorage中的参数
          setscene_id_ID(setFromRow_value);
          setFromRow(setFromRow_value);
          setFromRowSelect(setFromRow_value);
        }

        //做一些异步操作
        api.featureDisplaySetGetQuest(setFromRow_value.feature_set_id, true, false).then(item => {
          if (item.retcode === 0) {
            if (item.result.groupIsomerismFeatureInfo) {
              let isomerismFeatureInfoValue: any = [];
              for (let i = 0; i < item.result.groupIsomerismFeatureInfo.length; i++) {
                isomerismFeatureInfoValue = isomerismFeatureInfoValue.concat(
                  item.result.groupIsomerismFeatureInfo[i].isomerismFeatureInfoArray,
                );
              }

              setisomerismFeatureInfoValue(isomerismFeatureInfoValue);
              resolve(isomerismFeatureInfoValue);
            }
          } else if (item.retmsg) {
            message.error(`失败原因: ${item.retmsg}`);
          }
        });
      });
      return p;
    }
    isomerismFeatureInfoValueHandel().then(data => {
      const valueSum = { class: 'CONTEXTTYPE,HASHFUNC,FEATURESTYLE,FEATURETYPE' };
      let _data: any = data;
      api.featureKVDataDisplayPostQuest(valueSum).then(item => {
        if (item.retcode === 0) {
          const HASHFUNC = item.result.data.HASHFUNC;
          const CONTEXTTYPE = item.result.data.CONTEXTTYPE;
          const FEATURESTYLE = item.result.data.FEATURESTYLE;
          const FEATURETYPE = item.result.data.FEATURETYPE;

          setHASHFUNCValue(HASHFUNC);
          setCONTEXTTYPEValue(CONTEXTTYPE);
          setFEATURETYPE(FEATURETYPE);

          _data.map((items: any) => {
            return FEATURESTYLE.map((item: any) => {
              if (item.value === items.feature_type) {
                return setfeature_typeShow(item);
              }
            });
          });
          message.success('查询成功');
        } else if (item.retmsg) {
          message.error(`失败原因: ${item.retmsg}`);
        }
      });
    });

    function fe_proc_idHandle(fe_proc_id_ids: any) {
      api.featureFeProcConfigDetailDisplayGetQuest(fe_proc_id_ids.fe_proc_id).then(item => {
        if (item.retcode === 0) {
          // 基本信息
          const fe_proc_config = item.result.fe_proc_config;
          const input_item = item.result.input_item;
          const op_item = item.result.op_item;
          setfe_proc_config(fe_proc_config);
          const input_item_value = input_item.map((item: any) => {
            return { ...item, index: comStore.getUUid() };
          });
          setDatalist(input_item_value);
          const op_item_value = op_item.map((item: any) => {
            return { ...item, index: comStore.getUUid() };
          });
          setDataList2(op_item_value);
        } else if (item.retmsg) {
          message.error(`失败原因: ${item.retmsg}`);
        }
      });
    }

    let fe_proc_id_ids: any = {};
    if (history.location.state) {
      //  判断当前有参数
      fe_proc_id_ids = history.location.state;

      sessionStorage.setItem('featureConfigurationKeyID', JSON.stringify(fe_proc_id_ids)); // 存入到sessionStorage中
      setscene_id_ID(fe_proc_id_ids);
      if (fe_proc_id_ids.title === '更改特征插件配置') {
        fe_proc_idHandle(fe_proc_id_ids);
      }
    } else {
      fe_proc_id_ids = JSON.parse(sessionStorage.getItem('featureConfigurationKeyID') || ''); // 当state没有参数时，取sessionStorage中的参数
      setscene_id_ID(fe_proc_id_ids);
      if (fe_proc_id_ids.title === '更改特征插件配置') {
        fe_proc_idHandle(fe_proc_id_ids);
      }
    }
  }, []);
  function fe_proc_idHandle(fe_proc_id_ids: any) {
    api.featureFeProcConfigDetailDisplayGetQuest(fe_proc_id_ids.fe_proc_id).then(item => {
      if (item.retcode === 0) {
        // 基本信息
        const fe_proc_config = item.result.fe_proc_config;
        const input_item = item.result.input_item;
        const op_item = item.result.op_item;
        setfe_proc_config(fe_proc_config);
        const input_item_value = input_item.map((item: any) => {
          return { ...item, index: comStore.getUUid() };
        });
        setDatalist(input_item_value);
        const op_item_value = op_item.map((item: any) => {
          return { ...item, index: comStore.getUUid() };
        });
        setDataList2(op_item_value);
      } else if (item.retmsg) {
        message.error(`失败原因: ${item.retmsg}`);
      }
    });
  }
  //  表格中的修改One
  const showModalFuncModifyOne = (valueList: any) => {
    setvalueListTableOne(valueList);
    setVisibleOne(true);
  };
  const showModalFunc = () => {
    setVisibleOne(true);

    showModalFuncModifyOne({ title: '编辑输入特征配置', record: {} });
  };

  const statusOPTIONS = [
    { key: '1', value: 'Kge_Bkdr_Hash' },
    { key: '2', value: '测试发布' },
  ];

  const goToRegisterModelInformation = () => {
    const Input_item_value = {
      input_item: [
        ...Input_item.input_item.filter(item => {
          if (!item.cre_time && item.cre_time !== '') {
            return item;
          } else {
            return '';
          }
        }),
      ],
    };

    let Op_item_value = {};
    if (Op_item.opItems) {
      Op_item_value = {
        op_item: [
          ...Op_item.opItems.filter(item => {
            if (item.cre_time && item.cre_time !== '') {
              return item;
            } else {
              return '';
            }
          }),
        ],
      };
    } else {
      Op_item_value = {
        op_item: [
          ...Op_item.op_item.filter(item => {
            if (!item.cre_time && item.cre_time !== '') {
              return item;
            } else {
              return '';
            }
          }),
        ],
      };
    }

    if (scene_id_ID.title === '更改特征插件配置') {
      const FormTowSum = {
        ...fromRow,
        scene_id: scene_id_ID.scene_id,
      };
      const RegisterModelInformation = { ...FormTowSum, hash_func: fromRowSelect.hash_func };
      const RegisterModelInformationValue = { fe_proc_config: { ...RegisterModelInformation } };

      const RegisterModelInformationValueVlaue = Object.assign(RegisterModelInformationValue, Input_item_value);
      const RegisterModelInformationValueSum = Object.assign(RegisterModelInformationValueVlaue, Op_item_value);

      api.featureRegisterFeProcConfigQUEST(RegisterModelInformationValueSum).then(item => {
        if (item.retcode === 0) {
          if (scene_id_ID.title === '更改特征插件配置') {
            history.push('/HeterogeneousPlatform/Nationalkaraoke/SceneModelInformation');
          } else {
            history.push('/HeterogeneousPlatform/Nationalkaraoke/RegisterModelInformation');
          }
        } else if (item.retmsg) {
          message.error(`失败原因: ${item.retmsg}`);
        }
      });
    } else {
      let FormTowSum = {};
      if (fromRow.fe_proc_id && fromRow.fe_proc_id.fe_proc_id) {
        FormTowSum = {
          ...fromRow,
          scene_id: scene_id_ID.scene_id,
          fe_proc_id: fromRow.fe_proc_id.fe_proc_id,
        };
      } else {
        FormTowSum = {
          ...fromRow,
          scene_id: scene_id_ID.scene_id,
          // fe_proc_id: fromRow.fe_proc_id.fe_proc_id,
        };
      }

      const RegisterModelInformation = { ...FormTowSum, hash_func: fromRowSelect.hash_func };

      const RegisterModelInformationValue = { fe_proc_config: { ...RegisterModelInformation } };
      const RegisterModelInformationValueVlaue = Object.assign(RegisterModelInformationValue, Input_item_value);
      const RegisterModelInformationValueSum = Object.assign(RegisterModelInformationValueVlaue, Op_item_value);

      api.featureRegisterFeProcConfigQUEST(RegisterModelInformationValueSum).then(item => {
        if (item.retcode === 0) {
          if (scene_id_ID.title === '更改特征插件配置') {
            history.push('/HeterogeneousPlatform/Nationalkaraoke/SceneModelInformation');
          } else {
            history.push('/HeterogeneousPlatform/Nationalkaraoke/RegisterModelInformation');
          }
        } else if (item.retmsg) {
          message.error(`失败原因: ${item.retmsg}`);
        }
      });
    }
  };

  const setshowregister = (count: any) => {
    setOp_op_id(count);
  };
  const handleChangeInput_desc = (count: string) => {
    sethashIdValue(count);
  };
  return (
    <div className="FeatureConfigurationClass">
      <div className="bodyClass">
        <div className="SceneHeader">特征插件配置</div>
        <div className="site-card-border-less-wrapper">
          <Card title="基本信息" bordered={false}>
            <Row gutter={16}>
              <Col className="gutter-row" span={8}>
                <div style={style}>
                  <span style={{ width: '26%' }}>特征插件配置id</span>
                  <span>{fe_proc_config.fe_proc_id}</span>
                </div>
              </Col>
              <Col className="gutter-row" span={8}>
                <div style={style}>
                  <span style={{ width: '26%' }}>创建时间</span>
                  <span>{fe_proc_config.cre_time}</span>
                </div>
              </Col>
              <Col className="gutter-row" span={8}>
                <div style={style}>
                  <span style={{ width: '26%' }}>上次修改时间</span>
                  <span>{fe_proc_config.last_mod}</span>
                </div>
              </Col>
              <Col className="gutter-row" span={8}>
                <div style={style}>
                  <span style={{ width: '26%' }}>已绑定模型服务</span>
                  <span>{fe_proc_config.model_id}</span>
                </div>
              </Col>
              <Col className="gutter-row" span={8}>
                <div style={style}>
                  <span style={{ width: '26%' }}>配置英文名</span>
                  <Input
                    placeholder="请输入配置英文名"
                    style={{ width: 160 }}
                    value={fromRow.fe_proc_name}
                    onChange={fromFunChanger('fe_proc_name')}
                  />
                </div>
              </Col>
              <Col className="gutter-row" span={8}>
                <div style={style}>
                  <span style={{ width: '26%' }}>配置中文名</span>
                  <Input
                    placeholder="请输入配置中文名"
                    style={{ width: 160 }}
                    value={fromRow.fe_proc_desc}
                    onChange={fromFunChanger('fe_proc_desc')}
                  />
                </div>
              </Col>

              <Col className="gutter-row" span={8}>
                <div style={style}>
                  <span style={{ width: '26%' }}>责任人</span>
                  <Input
                    placeholder="请输入责任人"
                    style={{ width: 160 }}
                    value={fromRow.owner_rtxs}
                    onChange={fromFunChanger('owner_rtxs')}
                  />
                </div>
              </Col>
            </Row>
          </Card>

          <Card title="通用配置" bordered={false}>
            <Row gutter={16}>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>需要哈希</span>
                  <Radio.Group value={fromRow.need_hash} onChange={fromFunChanger('need_hash')}>
                    <Radio value={1}>是</Radio>
                    <Radio value={2}>否</Radio>
                  </Radio.Group>
                </div>
              </Col>

              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>哈希函数</span>
                  <Select
                    placeholder="请选择哈希函数"
                    value={fromRowSelect.hash_func}
                    onChange={fromFunChangerTwo('hash_func')}
                    style={{ width: '200px' }}
                  >
                    {HASHFUNCValue.map(item => {
                      return (
                        <Select.Option key={item.label} value={item.value}>
                          {item.label}
                        </Select.Option>
                      );
                    })}
                  </Select>
                </div>
              </Col>

              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>需要slot加1</span>
                  <Radio.Group value={fromRow.slot_flat} onChange={fromFunChanger('slot_flat')}>
                    <Radio value={1}>是</Radio>
                    <Radio value={2}>否</Radio>
                  </Radio.Group>
                </div>
              </Col>
            </Row>
          </Card>

          <Card
            title="输入特征配置"
            extra={
              <Button onClick={showModalFunc}>
                <PlusOutlined />
              </Button>
            }
            bordered={false}
          >
            <Form form={form} component={false}>
              <FeatureConfigurationTable columns={columns} data={Datalist} />
            </Form>

            <ModelOne
              CONTEXTTYPEValue={CONTEXTTYPEValue}
              handleChangeInput_descValue={handleChangeInput_desc}
              valueListTableOne={valueListTableOne}
              isomerismFeatureInfoValue={isomerismFeatureInfoValue}
              feature_typeShow={feature_typeShow}
              visibleOne={visibleOne}
              onCreateOne={onCreateOne}
              onCancel={() => {
                setVisibleOne(false);
              }}
            />
          </Card>

          <Card
            title="算子处理配置"
            extra={
              <Button onClick={showModalFuncTow}>
                <PlusOutlined />
              </Button>
            }
            bordered={false}
          >
            <Form form={form} component={false}>
              <FeatureConfigurationTableSort
                onDataSourceSort={(op_item: any) => {
                  setDataList2(op_item);
                }}
                columns={columnsSort}
                data={dataList2}
                onChange={setDataList2}
              />
            </Form>

            <CollectionCreateForm
              setshowregister={setshowregister}
              Datalist={Datalist}
              valueListTable={valueListTable}
              featureDisplayOpValue={featureDisplayOpValue}
              statusOPTIONS={statusOPTIONS}
              FEATURETYPE={FEATURETYPE}
              visible={visible}
              onCreate={onCreate}
              onCancel={() => {
                setVisible(false);
              }}
            />
          </Card>
          <div className="preservationClass">
            <Button
              style={{ backgroundColor: 'rgba(255, 87, 51, 1)', width: ' 140px', height: '40px', marginRight: '10px' }}
              onClick={() => history.go(-1)}
              type="primary"
            >
              返回
            </Button>
            <Button
              onClick={goToRegisterModelInformation}
              type="primary"
              style={{ backgroundColor: 'rgba(255, 87, 51, 1)', width: ' 140px', height: '40px' }}
            >
              {scene_id_ID.title === '更改特征插件配置' ? '修改' : '保存'}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
