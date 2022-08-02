import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Button, Input, Select, InputNumber, message } from 'antd';
import api from '@src/api';
import { useHistory } from 'react-router-dom';
import './RegisterFeaturePullService.css';
const style = { padding: '8px 0', display: 'flex', alignItems: 'center' };
export default function RegisterFeaturePullService() {
  const history = useHistory();
  const [fromRow, setFromRow] = useState({
      feature_serv_name: '',
      feature_serv_desc: '',
      bid: 0,
      owner_rtxs: '',
      target: '',
      timeout: 0,
    }),
    [DATASOURCESValue, setDATASOURCESValue] = useState([{ label: '', value: '' }]),
    [ROUTERSELECT, setROUTERSELECT] = useState([{ label: '', value: '' }]),
    [fromRowSelect, setFromRowSelect] = useState({
      namesapce: [],
      selector_name: [],
      region: [],
      feature_serv_type: [],
    }),
    [NAMESAPCE, setNAMESAPCE] = useState([{ label: '', value: '' }]),
    [Feature_set_idID, setFeature_set_idID] = useState(''),
    [REGION, setREGION] = useState([{ label: '', value: '' }]);

  function fromFunChanger(value: any) {
    return (e: any) => {
      if (value === 'timeout' || value === 'bid') {
        setFromRow({
          ...fromRow,
          [value]: e,
        });
      } else {
        setFromRow({
          ...fromRow,
          [value]: e.target.value,
        });
      }
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

  function RegisterFeaturePullServiceFunc() {
    const fromRowSum = {
      ...fromRow,
      feature_set_id: Feature_set_idID,
    };
    const RegisterFeaturePullServiceValue = Object.assign(fromRowSum, fromRowSelect);
    const RegisterFeaturePullServiceValueEnd = { ...RegisterFeaturePullServiceValue };
    api.featureRegisterFeatureServRouterPostQuest(RegisterFeaturePullServiceValueEnd).then((item: any) => {
      if (item.retcode === 0) {
        message.success('注册特征拉取服务路由成功');
        history.push({
          pathname: '/HeterogeneousPlatform/Nationalkaraoke/FeatureSetConfiguration',
          // state: item.result.data,
          state: history.location.state,
        });
      } else if (item.retmsg) {
        message.error(`失败原因: ${item.retmsg}`);
      }
    });
  }
  useEffect(() => {
    const valueSum = { class: 'NAMESAPCE,REGION,DATASOURCES,ROUTERSELECT' };

    api.featureKVDataDisplayPostQuest(valueSum).then(item => {
      if (item.retcode === 0) {
        const NAMESAPCE = item.result.data.NAMESAPCE;
        const REGION = item.result.data.REGION;
        const DATASOURCES = item.result.data.DATASOURCES;

        const ROUTERSELECT = item.result.data.ROUTERSELECT;
        setNAMESAPCE(NAMESAPCE);
        setREGION(REGION);
        setDATASOURCESValue(DATASOURCES);
        setROUTERSELECT(ROUTERSELECT);
        message.success('查询成功');
      } else if (item.retmsg) {
        message.error(`失败原因: ${item.retmsg}`);
      }
    });

    let setFromRow_value: any = {};

    if (history.location.state) {
      //  判断当前有参数
      setFromRow_value = history.location.state;
      sessionStorage.setItem('keyIDRegisterFeaturePullService', JSON.stringify(setFromRow_value)); // 存入到sessionStorage中
      setFeature_set_idID(setFromRow_value.sample_id);
    } else {
      setFromRow_value = JSON.parse(sessionStorage.getItem('keyIDRegisterFeaturePullService') || ''); // 当state没有参数时，取sessionStorage中的参数
      setFeature_set_idID(setFromRow_value.sample_id);
    }
  }, []);
  return (
    <div className="SceneRegistrationClass">
      <div className="bodyClass">
        <div className="SceneHeader">注册特征存储拉取服务</div>
        <div className="site-card-border-less-wrapper">
          <Card title="基本信息" bordered={false}>
            <Row gutter={16}>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>服务英文名</span>
                  <Input
                    placeholder="请输入服务英文名"
                    style={{ width: 200 }}
                    value={fromRow.feature_serv_name}
                    onChange={fromFunChanger('feature_serv_name')}
                  />
                </div>
              </Col>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>服务中文名</span>
                  <Input
                    placeholder="请输入服务中文名"
                    style={{ width: 200 }}
                    value={fromRow.feature_serv_desc}
                    onChange={fromFunChanger('feature_serv_desc')}
                  />
                </div>
              </Col>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>服务来源类型</span>
                  <Select
                    placeholder="请选择数据来源"
                    value={fromRowSelect.feature_serv_type}
                    onChange={fromFunChangerTwo('feature_serv_type')}
                    style={{ width: '200px' }}
                  >
                    {DATASOURCESValue.map(item => {
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
                  <span style={{ width: '26%' }}>bid</span>

                  <InputNumber
                    placeholder="请输入bid"
                    style={{ width: 200 }}
                    value={fromRow.bid}
                    onChange={fromFunChanger('bid')}
                  />
                </div>
              </Col>

              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>责任人</span>
                  <Input
                    placeholder="请输入责任人"
                    style={{ width: 200 }}
                    value={fromRow.owner_rtxs}
                    onChange={fromFunChanger('owner_rtxs')}
                  />
                </div>
              </Col>
            </Row>
          </Card>

          <Card title="绑定路由信息" bordered={false}>
            <Row gutter={16}>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>路由选择器</span>
                  <Select
                    placeholder="请选择路由选择器"
                    value={fromRowSelect.selector_name}
                    onChange={fromFunChangerTwo('selector_name')}
                    style={{ width: '200px' }}
                  >
                    {ROUTERSELECT.map(item => {
                      return (
                        <Select.Option key={item.value} value={item.value}>
                          {item.label}
                        </Select.Option>
                      );
                    })}
                  </Select>
                </div>
              </Col>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>命名空间</span>
                  <Select
                    placeholder="请选择命名空间"
                    value={fromRowSelect.namesapce}
                    onChange={fromFunChangerTwo('namesapce')}
                    style={{ width: '200px' }}
                  >
                    {NAMESAPCE.map(item => {
                      return (
                        <Select.Option key={item.value} value={item.value}>
                          {item.label}
                        </Select.Option>
                      );
                    })}
                  </Select>
                </div>
              </Col>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>地区</span>
                  <Select
                    placeholder="请选择地区"
                    value={fromRowSelect.region}
                    onChange={fromFunChangerTwo('region')}
                    style={{ width: '200px' }}
                  >
                    {REGION.map(item => {
                      return (
                        <Select.Option value={item.value} key={item.label}>
                          {item.label}
                        </Select.Option>
                      );
                    })}
                  </Select>
                </div>
              </Col>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>服务名字</span>
                  <Input
                    placeholder="请输入目标字段"
                    style={{ width: 200 }}
                    value={fromRow.target}
                    onChange={fromFunChanger('target')}
                  />
                </div>
              </Col>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>服务超时</span>

                  <InputNumber
                    addonAfter="ms"
                    placeholder="请输入服务超时"
                    style={{ width: 200 }}
                    value={fromRow.timeout}
                    onChange={fromFunChanger('timeout')}
                  />
                </div>
              </Col>
            </Row>
          </Card>
          <Button type="primary" className="preservationClass" onClick={RegisterFeaturePullServiceFunc}>
            保存
          </Button>
        </div>
      </div>
    </div>
  );
}
