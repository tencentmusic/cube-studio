import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Button, Space, message, Tooltip } from 'antd';
import { PlusOutlined } from '@ant-design/icons';
import SceneModeTable from '../SceneModeTable/SceneModeTable';
import SceneModeTableJH from '../SceneModeTable/SceneModeTableJH';
import { useHistory } from 'react-router-dom';
import api from '@src/api';
import './SceneModelInformation.css';
const style = { padding: '8px 0', display: 'flex', alignItems: 'center' };

export default function SceneModelInformation() {
  const history = useHistory();

  const [BasicData, setBasicData] = useState<any>({
      bid: '',
      business: '',
      cre_time: '',
      last_mod: '',
      owner_rtxs: '',
      report_field: '',
      resource_info: '',
      sample_id: '',
      scene_desc: '',
      scene_id: '',
      scene_name: '',
      tid: '',
      ReFeatureInformationTitle: '场景模型配置信息',
    }),
    [ModelConfigs, setModelConfigs] = useState([]),
    [FeatureSet, setFeatureSet] = useState([{ feature_set_id: '' }]),
    [feature_set_id, setFeature_set_id] = useState('');

  const columnsValue = [
    {
      title: '模型id',
      dataIndex: 'model_id',
      key: 'model_id',
      render: (text: any) => <div>{text}</div>,
    },
    {
      title: '名字',
      dataIndex: 'model_name',
      key: 'model_name',
      ellipsis: true,
      // ellipsis: {
      //   showTitle: false,
      // },
    },
    {
      title: '预测目标',
      dataIndex: 'preds',
      key: 'preds',
    },
    {
      title: '超时',
      dataIndex: 'resource_info',
      key: 'resource_info',
      render: (text: any) => {
        return text.timeout;
      },
    },
    {
      title: '特征插件配置',
      dataIndex: 'fe_proc_id',
      key: 'fe_proc_id',
      width: 120,
      render: (text: any) => {
        return (
          <Space size="middle">
            <a onClick={() => goWhereFeatureConfiguration(text)}>{text.fe_proc_name}</a>
          </Space>
        );
      },
    },
    {
      title: '模型服务配置',
      dataIndex: 'resource_info',
      key: 'resource_info',
      width: 120,
      ellipsis: true,
      render: (text: any) => {
        return (
          <Tooltip title={text.model_serv_name}>
            <Space size="middle">
              <a onClick={() => model_serv_nameConfiguration(text)}>{text.model_serv_name}</a>
            </Space>
          </Tooltip>
        );
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
      width: 140,
    },
    {
      title: '上次修改时间',
      dataIndex: 'last_mod',
      key: 'last_mod',
      width: 140,
    },
    {
      title: '操作',
      key: 'action',
      render: (text: any, record: any) => (
        <Space size="middle" onClick={() => goWhereOne(text)}>
          <a>Edit</a>
        </Space>
      ),
    },
  ];
  const columnsValueJH = [
    {
      title: '特征集合ID',
      dataIndex: 'feature_set_id',
      key: 'feature_set_id',
      // render: (text: any) => <a>{text}</a>,
    },
    {
      title: '名字',
      dataIndex: 'feature_set_name',
      key: 'feature_set_name',
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
      width: 140,
    },
    {
      title: '上次修改时间',
      dataIndex: 'last_mod',
      key: 'last_mod',
      width: 140,
    },

    {
      title: '操作',
      key: 'action',
      render: (text: any, record: any) =>
        BasicData.ReFeatureInformationTitle === '查看场景模型配置信息' ? (
          ''
        ) : (
          <Space size="middle" onClick={() => goWhereTwo(record)}>
            <a>Edit</a>
          </Space>
        ),
    },
  ];
  useEffect(() => {
    let setFromRow_value: any = {};

    if (history.location.state) {
      //  判断当前有参数
      setFromRow_value = history.location.state;
      sessionStorage.setItem('sceneModelInformationKeyID', JSON.stringify(setFromRow_value)); // 存入到sessionStorage中
      setBasicData(setFromRow_value);

      featureSceneDisplayGetQuestFun(setFromRow_value.scene_id);
    } else {
      setFromRow_value = JSON.parse(sessionStorage.getItem('sceneModelInformationKeyID') || ''); // 当state没有参数时，取sessionStorage中的参数

      setBasicData(setFromRow_value);

      featureSceneDisplayGetQuestFun(setFromRow_value.scene_id);
    }
  }, []);
  const goWhere = () => {
    history.push({
      pathname: '/HeterogeneousPlatform/Nationalkaraoke/RegisterModelInformation',
      state: { ...BasicData, feature_set_id: feature_set_id, title: '更改特征插件配置' },
    });
  };
  const goWhereTwo = (record: any) => {
    history.push({
      pathname: '/HeterogeneousPlatform/Nationalkaraoke/FeatureSetConfiguration',
      state: { ...BasicData, title: '更改特征集合配置', record },
    });
  };
  const goWhereOne = (testValue: any) => {
    history.push({
      pathname: '/HeterogeneousPlatform/Nationalkaraoke/RegisterModelInformation',
      state: { ...testValue, feature_set_id: feature_set_id, title: '更改注册模型信息' },
    });
  };
  const goWhereFeatureConfiguration = (testValue: any) => {
    history.push({
      pathname: '/HeterogeneousPlatform/Nationalkaraoke/FeatureConfiguration',
      state: { ...testValue, feature_set_id: feature_set_id, title: '更改特征插件配置' },
    });
  };
  const model_serv_nameConfiguration = (testValue: any) => {
    history.push({
      pathname: '/HeterogeneousPlatform/Nationalkaraoke/RegisterModelService',
      state: { ...testValue, feature_set_id: feature_set_id, title: '更改模型服务' },
    });
  };
  const SceneModelInformationHandle = () => {
    if (BasicData.ReFeatureInformationTitle === '修改场景模型配置信息') {
      featureSceneDisplayGetQuestFun(BasicData.scene_id);
      history.push('/HeterogeneousPlatform/Nationalkaraoke');
    } else if (BasicData.ReFeatureInformationTitle === '场景模型配置信息') {
      featureSceneDisplayGetQuestFun(BasicData.scene_id);
      history.push('/HeterogeneousPlatform/Nationalkaraoke');
    } else {
      history.push('/HeterogeneousPlatform/Nationalkaraoke');
    }
  };
  function featureSceneDisplayGetQuestFun(IDvalue: any) {
    api.featureSceneDisplayGetQuest(IDvalue).then(item => {
      if (item.retcode === 0) {
        message.success('查询成功');
        setModelConfigs(item.result.ModelConfigs);
        const FeatureSetArray = [];
        FeatureSetArray.push(item.result.FeatureSet);

        // sessionStorage.setItem('feature_set_id', JSON.stringify(item.result.FeatureSet.feature_set_id));
        setFeatureSet(FeatureSetArray as any);
        setFeature_set_id(item.result.FeatureSet.feature_set_id);
      } else if (item.retmsg) {
        message.error(`失败原因: ${item.retmsg}`);
      }
    });
  }
  const businessHandle = (business: number) => {
    switch (business) {
      case 1:
        return 'QQ音乐';
      case 2:
        return '全民K歌';
      default:
        return '未知';
    }
  };
  return (
    <div className="SceneModelInformationClassLess">
      <div className="bodyClass">
        <div className="SceneHeader">{BasicData?.ReFeatureInformationTitle}</div>
        <div className="site-card-border-less-wrapper">
          <Card title="基本信息" bordered={false}>
            <Row gutter={16}>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>场景ID</span>
                  <span>{BasicData?.scene_id}</span>
                </div>
              </Col>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>所属业务</span>
                  <span>{businessHandle(BasicData?.business)}</span>
                </div>
              </Col>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>场景英文名</span>
                  <span>{BasicData?.scene_name}</span>
                </div>
              </Col>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>责任人</span>
                  <span>{BasicData?.owner_rtxs}</span>
                </div>
              </Col>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>上报bid</span>
                  <span>{BasicData?.bid}</span>
                </div>
              </Col>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>上报tid</span>
                  <span>{BasicData?.tid}</span>
                </div>
              </Col>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>上报字段</span>
                  <span>{BasicData?.report_field}</span>
                </div>
              </Col>
            </Row>
          </Card>

          <Card
            title="特征集合配置信息"
            extra={FeatureSet.map(item => {
              return item.feature_set_id ? (
                ''
              ) : (
                <Button>
                  <PlusOutlined />
                </Button>
              );
            })}
            bordered={false}
          >
            <SceneModeTableJH columns={columnsValueJH} data={FeatureSet} />
          </Card>

          <Card
            title="模型服务配置信息"
            extra={
              BasicData.ReFeatureInformationTitle === '查看场景模型配置信息' ? (
                ''
              ) : (
                <Button onClick={goWhere}>
                  <PlusOutlined />
                </Button>
              )
            }
            bordered={false}
          >
            <SceneModeTable columns={columnsValue} data={ModelConfigs} />
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
              onClick={SceneModelInformationHandle}
            >
              保存
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
