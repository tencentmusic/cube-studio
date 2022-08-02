import React, { useState, useEffect } from 'react';
import { message, Row, Col, Button, Input, Card, Table, Modal } from 'antd';
const style = { padding: '8px 0', display: 'flex', alignItems: 'center' };
import api from '@src/api';
import { useHistory, useParams } from 'react-router-dom';
import './index.css';

interface Item {
  scene_id: string;
  name: string;
  age: number;
  address: string;
}

export default function index() {
  const history = useHistory();
  const routeParams = useParams();
  const [data, setData] = useState({ Total: 0, Scenes: [] });
  const [record, setRecord] = useState({ scene_id: undefined });
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [business, setBusiness] = useState(Number);
  const [P, setP] = useState(1);
  const [S, setS] = useState(10);

  const [fromRow, setFromRow] = useState({
    strategy_ids: [],
    component_marks: [],
    admins: [],
    versionDesc: '',
  });

  // 封装高阶函数收集input数据
  const fromFunChanger = (value: string) => {
    return (e: any) => {
      let target_value = null;
      if (value === 'versionDesc') {
        target_value = e.target.value;
      } else {
        target_value = e.target.value.split(';').splice('');
      }
      setFromRow({
        ...fromRow,
        [value]: target_value,
      });
    };
  };
  useEffect(() => {
    let setFromRow_value: any = {};
    console.log(history.location, '888', routeParams);
    const search_value = history.location.search.split('?')[history.location.search.split('?').length - 1];

    if (history.location.state) {
      //  判断当前有参数
      setFromRow_value = history.location.state || search_value;
      sessionStorage.setItem('NationalkaraokeKeyID', JSON.stringify(setFromRow_value)); // 存入到sessionStorage中
      console.log(setFromRow_value, '11111111111');
      setBusiness(setFromRow_value);
      featureScenePagesDisplayPostQuestFun(P, S, fromRow.strategy_ids, setFromRow_value);
    } else {
      setFromRow_value = JSON.parse(sessionStorage.getItem('NationalkaraokeKeyID') || search_value); // 当state没有参数时，取sessionStorage中的参数
      console.log(setFromRow_value, '1111112222222222');
      setBusiness(setFromRow_value);
      featureScenePagesDisplayPostQuestFun(P, S, fromRow.strategy_ids, setFromRow_value);
    }
  }, []);
  function featureScenePagesDisplayPostQuestFun(P: number, S: number, owner_rtxs: any, business: number) {
    api.featureScenePagesDisplayPostQuest(P, S, owner_rtxs, business).then((item: any) => {
      if (item.retcode === 0) {
        setData(item.result);
        message.success('查询成功');
      } else if (item.retmsg) {
        message.error(`失败原因: ${item.retmsg}`);
      }
    });
  }

  //  查看表格
  const goWhereReFeatureInformationTwoSee = (rowValue: any) => {
    history.push({
      pathname: '/HeterogeneousPlatform/Nationalkaraoke/SceneModelInformation',
      state: { ...rowValue, ReFeatureInformationTitle: '查看场景模型配置信息' },
    });
  };
  //  修改表格
  const goWhereReFeatureInformationTwoSeeTwo = (rowValue: any) => {
    console.log(rowValue, '1111111');

    history.push({
      pathname: '/HeterogeneousPlatform/Nationalkaraoke/SceneModelInformation',
      state: { ...rowValue, ReFeatureInformationTitle: '修改场景模型配置信息' },
    });
  };
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
  const columns = [
    {
      title: '场景id',
      dataIndex: 'scene_id',
      width: '80px',

      render: (text: any, record: any) => (
        <Button onClick={() => goWhereReFeatureInformationTwoSee(record)} type="link">
          {text}
        </Button>
      ),
    },
    {
      title: '场景英文名',
      dataIndex: 'scene_name',
    },
    {
      title: '场景中文名',
      dataIndex: 'scene_desc',
    },
    {
      title: '特征上报字段',
      dataIndex: 'report_field',
    },
    {
      title: '特征上报bid',
      dataIndex: 'bid',
    },
    {
      title: '特征上报tid',
      dataIndex: 'tid',
    },
    {
      title: '责任人',
      dataIndex: 'owner_rtxs',
    },
    {
      title: '创建时间',
      dataIndex: 'cre_time',
      width: '110px',
    },
    {
      title: '修改时间',
      dataIndex: 'last_mod',
      width: '110px',
    },

    {
      title: '所属业务',
      dataIndex: 'business',
      render: (_: any, record: Item) => {
        return businessHandle(_);
      },
    },

    {
      title: '模型服务路由id',
      dataIndex: 'resource_info',
    },
    {
      title: '特征集合（样本）的id',
      dataIndex: 'sample_id',
      width: 100,
    },

    {
      title: 'operation',
      dataIndex: 'operation',
      render: (_: any, record: Item) => {
        return (
          <div style={{ display: 'flex', justifyContent: 'space-around' }}>
            <Button type="link" onClick={() => goWhereReFeatureInformationTwoSeeTwo(record)}>
              编辑
            </Button>
            <Button type="link" onClick={() => setIsModalVisibleShow(record)}>
              生成配置
            </Button>
            <Button type="link" onClick={() => downloadVisible(record)}>
              下载配置
            </Button>
          </div>
        );
      },
    },
  ];
  // 查询
  const queryFunc = () => {
    featureScenePagesDisplayPostQuestFun(P, S, fromRow.strategy_ids, business);
  };
  // 新建
  const NewFunc = () => {
    history.push({
      pathname: '/HeterogeneousPlatform/Nationalkaraoke/SceneRegistration',
      state: history.location.state,
    });
  };
  function onChangeHandel(page: any, pageSize: any) {
    featureScenePagesDisplayPostQuestFun(page, pageSize, fromRow.strategy_ids, business);
  }

  const setIsModalVisibleShow = (record: any) => {
    setRecord(record);
    setIsModalVisible(true);
  };
  //  下载配置
  const downloadVisible = (record: any) => {
    window.open(`http://11.161.238.209:8080/api/featureDownConfiguration?sceneId=${record.scene_id}`);
  };
  const handleOk = () => {
    if (fromRow.versionDesc === '') {
      return message.error(`填写版本信息`);
    } else {
      api.featureBuildConfigurationGetQuest(record.scene_id, fromRow.versionDesc).then(item => {
        if (item.retcode === 0) {
          message.success('生成配置成功');
          setIsModalVisible(false);
        } else if (item.retmsg) {
          message.error(`失败原因: ${item.retmsg}`);
        }
      });
    }
  };

  const handleCancel = () => {
    setIsModalVisible(false);
  };
  return (
    <div>
      <div className="site-card-border-less-wrapper">
        <Card bordered={false}>
          <Row gutter={16}>
            <Col className="gutter-row" span={6}>
              <div style={style}>
                <span style={{ width: '20%' }}>责任人</span>
                <Input
                  placeholder="请输入策略ID"
                  value={fromRow.strategy_ids}
                  onChange={fromFunChanger('strategy_ids')}
                />
              </div>
            </Col>
            {/* <Col className="gutter-row" span={6}>
              <div style={style}>
                <span style={{ width: '10%' }}>组件</span>
                <Input
                  placeholder="请输入组件mark"
                  value={fromRow.component_marks}
                  onChange={fromFunChanger('component_marks')}
                />
              </div>
            </Col> */}
            <Col className="gutter-row2" span={4}>
              <Button type="primary" style={{ marginTop: '8px' }} onClick={queryFunc}>
                查询
              </Button>
            </Col>
            <Col className="gutter-row2" span={4} offset={10}>
              <Button type="primary" style={{ marginTop: '8px' }} onClick={NewFunc}>
                新建
              </Button>
            </Col>
          </Row>
        </Card>

        <Card bordered={false} className="cardTwo">
          <Table
            scroll={{ x: '100%' }}
            pagination={{
              total: data.Total,
              onChange: (page, pageSize) => onChangeHandel(page, pageSize),
            }}
            bordered
            dataSource={data.Scenes}
            columns={columns}
            rowClassName="editable-row"
          />
        </Card>

        <Modal
          okText="确认"
          cancelText="取消"
          title="填写版本信息"
          visible={isModalVisible}
          onOk={handleOk}
          onCancel={handleCancel}
        >
          <Input value={fromRow.versionDesc} onChange={fromFunChanger('versionDesc')} placeholder="请填写版本信息" />
        </Modal>
      </div>
    </div>
  );
}
