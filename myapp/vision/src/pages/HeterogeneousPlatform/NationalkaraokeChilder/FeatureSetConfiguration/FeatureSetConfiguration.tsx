import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Button, Input, message, Space, Tooltip } from 'antd';
import { PlusOutlined } from '@ant-design/icons';
import { useHistory } from 'react-router-dom';
import SceneModeTable from '../SceneModeTable/FeatureSetConfigurationTbale';
import SceneModeTableJH from '../SceneModeTable/SceneModeTableJH';
import SceneModeTableJHChidren from '../SceneModeTable/SceneModeTableJHChidren';
// import VirtualTable from '../SceneModeTable/VirtualTable';
import CollectionCreateForm from './FeatureSetConfigurationModel';
import api from '@src/api';
import './FeatureSetConfiguration.css';
const style = { padding: '8px 0', display: 'flex', alignItems: 'center' };

export default function FeatureSetConfiguration() {
  const history = useHistory();
  const [fromRow, setFromRow] = useState({
      sample_id: '',
      strategy_ids: '',
      owner_rtxs: '',
      feature_set_desc: '',
    }),
    [featureServRouter, setfeatureServRouter] = useState([]),
    [feature_set_id, setFeature_set_id] = useState({ sample_id: '' }),
    [Feature_set_idID, setFeature_set_idID] = useState({ Feature_set_idID: '' }),
    [visible, setVisible] = useState(false),
    [isomerismFeatureInfo, setIsomerismFeatureInfo] = useState([]),
    [VisibleSee_title, setVisibleSee_title] = useState({
      title: '查看特征配置',
      record: {},
    });
  function fromFunChanger(value: any) {
    return (e: any) => {
      setFromRow({
        ...fromRow,
        [value]: e.target.value,
      });
    };
  }

  function setFromRow_valueHandele(setFromRow_value: any) {
    if (setFromRow_value.record && setFromRow_value.title === '更改特征集合配置') {
      fromRow.sample_id = setFromRow_value.record.feature_set_id;
      fromRow.strategy_ids = setFromRow_value.record.feature_set_name;
      fromRow.owner_rtxs = setFromRow_value.record.owner_rtxs;
      fromRow.feature_set_desc = setFromRow_value.record.feature_set_desc;
    } else {
      fromRow.sample_id = setFromRow_value.sample_id;
      fromRow.strategy_ids = setFromRow_value.scene_name;
      fromRow.owner_rtxs = setFromRow_value.owner_rtxs;
      fromRow.feature_set_desc = setFromRow_value.scene_desc;
    }
    api.featureDisplaySetGetQuest(fromRow.sample_id, true, true).then(item => {
      if (item.retcode === 0) {
        const featureServRouter = item.result.featureServRouter;
        const isomerismFeatureInfo = item.result.groupIsomerismFeatureInfo;

        setfeatureServRouter(featureServRouter);
        setIsomerismFeatureInfo(isomerismFeatureInfo);
      } else if (item.retmsg) {
        message.error(`失败原因: ${item.retmsg}`);
      }
    });
  }
  useEffect(() => {
    let setFromRow_value: any = {};

    if (history.location.state) {
      //  判断当前有参数
      setFromRow_value = history.location.state;
      sessionStorage.setItem('keyIDFeatureSetConfiguration', JSON.stringify(setFromRow_value)); // 存入到sessionStorage中

      setFromRow_valueHandele(setFromRow_value);
      setFeature_set_idID(setFromRow_value);
      setFeature_set_id(setFromRow_value);
    } else {
      setFromRow_value = JSON.parse(sessionStorage.getItem('keyIDFeatureSetConfiguration') || ''); // 当state没有参数时，取sessionStorage中的参数

      setFromRow_valueHandele(setFromRow_value);

      setFeature_set_idID(setFromRow_value);
      setFeature_set_id(setFromRow_value);
    }
  }, []);

  const source_typeHanle = (value: string) => {
    switch (value) {
      case '4':
        return '统一特征';
      case '5':
        return '异构特征';
      default:
        return '未知';
    }
  };
  const columnsValueJH = [
    {
      title: '特征组id',
      dataIndex: 'group_id',
      key: 'group_id',
      render: (text: any, record: any) => (
        <Button onClick={() => goWhereReFeatureInformationTwoSee(record)} type="link">
          {text}
        </Button>
      ),
    },
    {
      title: '特征组类型',
      dataIndex: 'isomerismFeatureInfoArray',
      key: 'group_id',
      render: (text: any, record: any) => {
        return record.isomerismFeatureInfoArray.length;
      },
    },
    {
      title: '特征配置',
      dataIndex: 'source_type',
      key: 'source_type',
      render: (text: any, record: any) => {
        return source_typeHanle(text);
      },
    },
    {
      title: '特征组名称',
      dataIndex: 'feature_group_name',
      key: 'feature_group_name',
    },
    {
      title: '特征组描述',
      dataIndex: 'feature_group_desc',
      key: 'feature_group_desc',
    },

    {
      title: '责任人',
      dataIndex: 'owner_rtxs',
      key: 'owner_rtxs',
    },
    {
      title: '特征字段详情',
      dataIndex: 'isomerismFeatureInfoArray',
      key: 'group_id',
      width: 500,
      render: (text: any, record: any) => {
        const columnsValueJH_children = [
          {
            title: '特征字段ID',
            dataIndex: 'feature_id',
            key: 'feature_id',
          },
          {
            title: '特征字段名',
            dataIndex: 'unique_feature_name',
            key: 'unique_feature_name',
            ellipsis: {
              showTitle: false,
            },
            render: (unique_feature_name: string) => (
              <Tooltip placement="topLeft" title={unique_feature_name}>
                {unique_feature_name}
              </Tooltip>
            ),
          },
          {
            title: '唯一特征名',
            dataIndex: 'feature_name',
            key: 'feature_name',
            ellipsis: {
              showTitle: false,
            },
            render: (feature_name: string) => (
              <Tooltip placement="topLeft" title={feature_name}>
                {feature_name}
              </Tooltip>
            ),
          },
          {
            title: '特征哈希ID',
            dataIndex: 'hashId',
            key: 'hashId',
          },
          {
            title: '特征字段描述',
            dataIndex: 'feature_desc',
            key: 'feature_desc',
            ellipsis: {
              showTitle: false,
            },
            render: (feature_desc: string) => (
              <Tooltip placement="topLeft" title={feature_desc}>
                {feature_desc}
              </Tooltip>
            ),
          },
        ];
        return <SceneModeTableJHChidren columns={columnsValueJH_children} data={record.isomerismFeatureInfoArray} />;
      },
    },

    {
      title: 'Action',
      key: 'action',
      render: (text: any, record: any) => (
        <>
          <Space size="middle">
            <Button size="middle" onClick={() => goWhereReFeatureInformationTwo(record)} type="link">
              Edit
            </Button>
          </Space>

          <Space size="middle">
            <Button size="middle" onClick={() => deleteFuncReFeatureInformationTwo(record)} type="link">
              delete
            </Button>
          </Space>
        </>
      ),
    },
  ];
  function feature_serv_type_tableValue(text: any) {
    switch (text) {
      case '4':
        return '统一特征';
      case '5':
        return '异构特征';
      default:
        return '其他';
    }
  }
  const columnsValue = [
    {
      title: 'id',
      dataIndex: 'feature_serv_id',
      key: 'feature_serv_id',
      width: 60,
      render: (text: any, record: any) => (
        <Button onClick={() => setVisibleShowSee({ record })} type="link">
          {text}
        </Button>
      ),
    },
    {
      title: '服务名字',
      dataIndex: 'feature_serv_name',
      key: 'feature_serv_name',
      width: 100,
    },
    {
      title: '服务类型',
      dataIndex: 'feature_serv_type',
      key: 'feature_serv_type',
      width: 100,
      render: (text: any) => {
        return feature_serv_type_tableValue(text);
      },
    },
    {
      title: '责任人',
      dataIndex: 'owner_rtxs',
      key: 'owner_rtxs',
    },
    {
      title: '创建时间',
      dataIndex: 'cre_time',
      key: 'cre_time',
      width: 120,
    },
    {
      title: '上次修改时间',
      dataIndex: 'last_mod',
      key: 'last_mod',
      width: 120,
    },

    {
      title: '操作',
      key: 'action',
      render: (text: any, record: any) => (
        <>
          <Space size="middle">
            <Button type="link" onClick={() => showModalFuncModify({ title: '修改路由配置', record })}>
              Edit
            </Button>
          </Space>
          <Space size="middle">
            <Button size="middle" onClick={() => deleteFuncModify(record)} type="link">
              delete
            </Button>
          </Space>
        </>
      ),
    },
  ];

  const goWhereSceneModelInformation = () => {
    let dataValue = {
      feature_set_id: Number(feature_set_id.sample_id),
    };

    if (fromRow.strategy_ids) {
      dataValue = Object.assign(dataValue, { feature_set_name: fromRow.strategy_ids });
    }
    if (fromRow.feature_set_desc) {
      dataValue = Object.assign(dataValue, { feature_set_desc: fromRow.feature_set_desc });
    }
    if (fromRow.owner_rtxs) {
      dataValue = Object.assign(dataValue, { owner_rtxs: fromRow.owner_rtxs });
    }

    api.featureUpdateSetPostQuest(dataValue).then(item => {
      if (item.retcode === 0) {
        history.push({
          pathname: '/HeterogeneousPlatform/Nationalkaraoke/SceneModelInformation',
          state: history.location.state,
        });
      } else if (item.retmsg) {
        message.error(`失败原因: ${item.retmsg}`);
      }
    });
  };
  const goWhereRegisterFeaturePullService = () => {
    history.push({
      pathname: '/HeterogeneousPlatform/Nationalkaraoke/RegisterFeaturePullService',
      state: { ...Feature_set_idID },
    });
  };
  //  新建特征配置
  const goWhereReFeatureInformation = () => {
    history.push({
      pathname: '/HeterogeneousPlatform/Nationalkaraoke/ReFeatureInformation',
      state: { ...Feature_set_idID, ReFeatureInformationTitle: '注册特征信息' },
    });
  };

  //  修改更新第二个表格
  const goWhereReFeatureInformationTwo = (rowValue: any) => {
    history.push({
      pathname: '/HeterogeneousPlatform/Nationalkaraoke/ReFeatureInformation',
      state: { ...rowValue, ReFeatureInformationTitle: '修改注册特征信息' },
    });
  };
  //  删除更新第一个表格
  const deleteFuncModify = (rowValue: any) => {
    const _rowValue = {
      feature_serv_id: rowValue.feature_serv_id,
      isDel: 1,
    };
    api.featureRegisterFeatureServRouterPostQuest(_rowValue).then(item => {
      if (item.retcode === 0) {
        message.success('删除成功');
        setFromRow_valueHandele(Feature_set_idID);
      } else if (item.retmsg) {
        message.error(`删除失败原因: ${item.retmsg}`);
      }
    });
  };
  //  删除更新第二个表格
  const deleteFuncReFeatureInformationTwo = (rowValue: any) => {
    console.log(rowValue, '6666');
    const valueSum = {
      group_id: rowValue.group_id,
      isDel: 1,
    };
    api.featureUpdateIsomerismFeatureFeatureInfoQUEST(valueSum).then(item => {
      if (item.retcode === 0) {
        message.success('删除注册特征组信息');
        setFromRow_valueHandele(Feature_set_idID);
      } else if (item.retmsg) {
        message.error(`失败原因: ${item.retmsg}`);
      }
    });
  };

  //  查看第二个表格
  const goWhereReFeatureInformationTwoSee = (rowValue: any) => {
    history.push({
      pathname: '/HeterogeneousPlatform/Nationalkaraoke/ReFeatureInformation',
      state: { ...rowValue, ReFeatureInformationTitle: '查看注册特征信息' },
    });
  };

  //  查看、修改弹窗
  const onCreate = (values: any) => {
    if (VisibleSee_title.title === '修改路由配置') {
      const visibleSee_title_value = {
        ...values,
        feature_serv_id: Number(values.feature_serv_id),
      };
      api.featureRegisterFeatureServRouterPostQuest(visibleSee_title_value).then(item => {
        if (item.retcode === 0) {
          setVisible(false);
          message.success(`修改路由配置成功`);
          setFromRow_valueHandele(Feature_set_idID);
        } else if (item.retmsg) {
          message.error(`失败原因: ${item.retmsg}`);
        }
      });
    }
    setVisible(false);
  };

  function setVisibleShowSee(record: any) {
    setVisibleSee_title({ ...record, title: '查看特征配置' });
    setVisible(true);
  }
  function showModalFuncModify(valueList: any) {
    setVisibleSee_title(valueList);
    setVisible(true);
  }
  return (
    <div className="SceneModelInformationClass">
      <div className="bodyClass">
        <div className="SceneHeader">特征集合配置</div>
        <div className="site-card-border-less-wrapper">
          <Card title="基本信息" bordered={false}>
            <Row gutter={16}>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '30%' }}>sample_id</span>
                  <Input style={{ width: 160 }} value={fromRow.sample_id} disabled />
                </div>
              </Col>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '30%' }}>特征集合英文名</span>
                  <Input
                    placeholder="请输入特征集合英文名"
                    style={{ width: 160 }}
                    value={fromRow.strategy_ids}
                    onChange={fromFunChanger('strategy_ids')}
                  />
                </div>
              </Col>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '30%' }}>特征集合中文名</span>
                  <Input
                    placeholder="请输入特征集合中文名"
                    style={{ width: 160 }}
                    value={fromRow.feature_set_desc}
                    onChange={fromFunChanger('feature_set_desc')}
                  />
                </div>
              </Col>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '30%' }}>责任人</span>
                  <Input
                    placeholder="请输入责任人"
                    style={{ width: 160 }}
                    value={fromRow.owner_rtxs}
                    onChange={fromFunChanger('owner_rtxs')}
                  />
                </div>
              </Col>
            </Row>

            {/* <VirtualTable columns={columnsVirtualTable} dataSource={dataVirtualTable} scroll={{ y: 300, x: '100vw' }} /> */}
          </Card>

          <Card
            title="路由配置"
            extra={
              <Button onClick={goWhereRegisterFeaturePullService}>
                <PlusOutlined />
              </Button>
            }
            bordered={false}
          >
            <SceneModeTable columns={columnsValue} data={featureServRouter} />
          </Card>

          <Card
            title="特征配置"
            extra={
              <Button onClick={goWhereReFeatureInformation}>
                <PlusOutlined />
              </Button>
            }
            bordered={false}
          >
            <SceneModeTableJH columns={columnsValueJH} data={isomerismFeatureInfo} />
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
              style={{ backgroundColor: 'rgba(255, 87, 51, 1)', width: ' 140px', height: '40px' }}
              type="primary"
              onClick={goWhereSceneModelInformation}
            >
              保存
            </Button>
          </div>

          <CollectionCreateForm
            visible={visible}
            onCreate={onCreate}
            VisibleSee_title={VisibleSee_title}
            onCancel={() => {
              setVisible(false);
            }}
          />
        </div>
      </div>
    </div>
  );
}
