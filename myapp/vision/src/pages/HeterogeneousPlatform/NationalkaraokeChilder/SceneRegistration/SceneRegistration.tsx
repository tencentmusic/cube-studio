import React, { useState, useEffect } from 'react';
import api from '@src/api';
import { useHistory } from 'react-router-dom';
import { Card, Row, Col, Button, Input, message } from 'antd';
import './SceneRegistration.css';
const style = { padding: '8px 0', display: 'flex', alignItems: 'center' };
export default function SceneRegistration() {
  const history = useHistory();
  const [fromRow, setFromRow] = useState({
    scene_name: '',
    scene_desc: '',
    owner_rtxs: '',
    bid: '',
    tid: '',
    report_field: '',
  });
  const [businessID, setBusinessID] = useState('');
  function fromFunChanger(value: any) {
    return (e: any) => {
      setFromRow({
        ...fromRow,
        [value]: e.target.value,
      });
    };
  }
  function SceneRegistrationHandle() {
    const fromRowSum = {
      ...fromRow,
      business: businessID,
    };
    api.featureRegisterScenePostQuest(fromRowSum).then((item: any) => {
      if (item.retcode === 0) {
        message.success('场景注册成功');
        sessionStorage.setItem('keyID', JSON.stringify(item.result.data));
        history.push({
          pathname: '/HeterogeneousPlatform/Nationalkaraoke/SceneModelInformation',
          state: { ...item.result.data, ReFeatureInformationTitle: '场景模型配置信息' },
        });
        setFromRow({
          scene_name: '',
          scene_desc: '',
          owner_rtxs: '',
          bid: '',
          tid: '',
          report_field: '',
        });
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
      sessionStorage.setItem('NationalkaraokeKeyID', JSON.stringify(setFromRow_value)); // 存入到sessionStorage中
      setBusinessID(setFromRow_value);
    } else {
      setFromRow_value = JSON.parse(sessionStorage.getItem('NationalkaraokeKeyID') || ''); // 当state没有参数时，取sessionStorage中的参数
      setBusinessID(setFromRow_value);
    }
  }, []);
  return (
    <div className="SceneRegistrationClass">
      <div className="bodyClass">
        <div className="SceneHeader">场景注册</div>
        <div className="site-card-border-less-wrapper">
          <Card title="基本信息" bordered={false}>
            <Row gutter={16}>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>场景英文名</span>
                  <Input
                    placeholder="请输入场景英文名"
                    style={{ width: 160 }}
                    value={fromRow.scene_name}
                    onChange={fromFunChanger('scene_name')}
                  />
                </div>
              </Col>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>场景中文名</span>
                  <Input
                    placeholder="请输入场景中文名"
                    style={{ width: 160 }}
                    value={fromRow.scene_desc}
                    onChange={fromFunChanger('scene_desc')}
                  />
                </div>
              </Col>
              <Col className="gutter-row" span={12}>
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

          <Card title="场景上报信息" bordered={false}>
            <Row gutter={16}>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>bid</span>
                  <Input
                    placeholder="请输入bid"
                    style={{ width: 160 }}
                    value={fromRow.bid}
                    onChange={fromFunChanger('bid')}
                  />
                </div>
              </Col>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>tid</span>
                  <Input
                    placeholder="请输入tid"
                    style={{ width: 160 }}
                    value={fromRow.tid}
                    onChange={fromFunChanger('tid')}
                  />
                </div>
              </Col>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>上报字段</span>
                  <Input
                    placeholder="请输入上报字段"
                    style={{ width: 160 }}
                    value={fromRow.report_field}
                    onChange={fromFunChanger('report_field')}
                  />
                </div>
              </Col>
            </Row>
          </Card>
          <Button type="primary" className="preservationClass" onClick={SceneRegistrationHandle}>
            保存
          </Button>
        </div>
      </div>
    </div>
  );
}
