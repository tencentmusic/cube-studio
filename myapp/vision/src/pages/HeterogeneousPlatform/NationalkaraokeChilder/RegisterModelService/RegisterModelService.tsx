import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Button, Input, Select, InputNumber, message, Tooltip } from 'antd';
import api from '@src/api';
import { useHistory } from 'react-router-dom';
import './RegisterModelService.css';
const style = { padding: '8px 0', display: 'flex', alignItems: 'center' };
export default function RegisterModelService() {
  const history = useHistory();
  const [fromRow, setFromRow] = useState({
      model_serv_name: '',
      model_serv_desc: '',
      owner_rtxs: '',
      target: '',
      timeout: '',
    }),
    [fromRowSelect, setFromRowSelect] = useState({
      selector_name: [],
      namesapce: [],
      region: [],
    }),
    [NAMESAPCE, setNAMESAPCE] = useState([{ label: '', value: '' }]),
    [scene_id_ID, setscene_id_ID] = useState({ scene_id: '', title: '', model_serv_id: '' }),
    [ROUTERSELECT, setROUTERSELECT] = useState([{ label: '', value: '' }]),
    [REGION, setREGION] = useState([{ label: '', value: '' }]);
  function fromFunChanger(value: any) {
    return (e: any) => {
      if (value === 'timeout') {
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
  useEffect(() => {
    const valueSum = { class: 'NAMESAPCE,REGION,ROUTERSELECT' };

    api.featureKVDataDisplayPostQuest(valueSum).then(item => {
      if (item.retcode === 0) {
        console.log(item.result.data, '1111');
        const NAMESAPCE = item.result.data.NAMESAPCE;
        const REGION = item.result.data.REGION;
        const ROUTERSELECT = item.result.data.ROUTERSELECT;
        setNAMESAPCE(NAMESAPCE);
        setREGION(REGION);
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
      console.log(setFromRow_value, 'setFromRow_value111111111');

      sessionStorage.setItem('featureConfigurationKeyID', JSON.stringify(setFromRow_value)); // 存入到sessionStorage中
      setscene_id_ID(setFromRow_value);
      setFromRow(setFromRow_value);
      setFromRowSelect(setFromRow_value);
    } else {
      setFromRow_value = JSON.parse(sessionStorage.getItem('featureConfigurationKeyID') || ''); // 当state没有参数时，取sessionStorage中的参数
      console.log(setFromRow_value, '888');

      setscene_id_ID(setFromRow_value);
      setFromRow(setFromRow_value);
      setFromRowSelect(setFromRow_value);
    }
  }, []);

  const goToRegisterModelInformation = () => {
    const fromRowSum = {
      ...fromRow,
      scene_id: scene_id_ID.scene_id,
    };
    const RegisterModelInformationValue = Object.assign(fromRowSum, fromRowSelect);
    if (scene_id_ID.title === '更改模型服务') {
      const RegisterModelInformationValue = {
        ...fromRow,
        scene_id: scene_id_ID.scene_id,
        selector_name: fromRowSelect.selector_name,
        namesapce: fromRowSelect.namesapce,
        region: fromRowSelect.region,
        model_serv_id: scene_id_ID.model_serv_id,
      };
      // api.featureUpdateModelConfigPostQuest(RegisterModelInformationValue).then(item => {
      //   if (item.retcode === 0) {
      //     history.push('/HeterogeneousPlatform/Nationalkaraoke/SceneModelInformation');
      //   } else if (item.retmsg) {
      //     message.error(`失败原因: ${item.retmsg}`);
      //   }
      // });
      api.featureRegisterModelServRouterPostQuest(RegisterModelInformationValue).then(item => {
        if (item.retcode === 0) {
          message.success('注册模型服务成功');
          history.push('/HeterogeneousPlatform/Nationalkaraoke/SceneModelInformation');
        } else if (item.retmsg) {
          message.error(`失败原因: ${item.retmsg}`);
        }
      });
    } else {
      api.featureRegisterModelServRouterPostQuest(RegisterModelInformationValue).then(item => {
        if (item.retcode === 0) {
          message.success('注册模型服务成功');
          history.push('/HeterogeneousPlatform/Nationalkaraoke/RegisterModelInformation');
        } else if (item.retmsg) {
          message.error(`失败原因: ${item.retmsg}`);
        }
      });
    }
  };
  return (
    <div className="SceneRegistrationClass">
      <div className="bodyClass">
        <div className="SceneHeader">注册模型服务</div>
        <div className="site-card-border-less-wrapper">
          <Card title="基本信息" bordered={false}>
            <Row gutter={16}>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>服务英文名</span>
                  <Input
                    placeholder="请输入服务英文名"
                    style={{ width: 200 }}
                    value={fromRow.model_serv_name}
                    onChange={fromFunChanger('model_serv_name')}
                  />
                </div>
              </Col>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>服务中文名</span>
                  <Input
                    placeholder="请输入服务中文名"
                    style={{ width: 200 }}
                    value={fromRow.model_serv_desc}
                    onChange={fromFunChanger('model_serv_desc')}
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
                  <span style={{ width: '30%' }}>命名空间</span>
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
                  <Tooltip title="北极星服务地址" color="blue">
                    <span style={{ width: '30%', color: 'skyblue' }}>模型服务地址</span>
                  </Tooltip>

                  <Tooltip title={fromRow.target} color="yellow">
                    <Input
                      placeholder="请输入目标字段"
                      style={{ width: 200 }}
                      value={fromRow.target}
                      onChange={fromFunChanger('target')}
                    />
                  </Tooltip>
                </div>
              </Col>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>服务超时</span>

                  <InputNumber
                    value={fromRow.timeout}
                    onChange={fromFunChanger('timeout')}
                    addonAfter="ms"
                    placeholder="请输入服务超时"
                    style={{ width: 200 }}
                  />
                </div>
              </Col>
            </Row>
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
              onClick={goToRegisterModelInformation}
            >
              {scene_id_ID.title === '更改模型服务' ? '修改' : '保存'}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
