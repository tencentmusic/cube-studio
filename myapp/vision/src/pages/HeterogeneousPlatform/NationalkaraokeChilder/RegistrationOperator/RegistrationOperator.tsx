import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Button, Input, Select, message } from 'antd';
import api from '@src/api';
import './RegistrationOperator.css';
const style = { padding: '8px 0', display: 'flex', alignItems: 'center' };
export default function RegistrationOperator() {
  const [fromRow, setFromRow] = useState({
      op_name: '',
      op_desc: '',
      owner_rtxs: '',
    }),
    [fromRowSelect, setFromRowSelect] = useState({
      param_count: [],
      param_type1: [],
      param_type2: [],
      param_type3: [],
      param_type4: [],
      param_type5: [],
      param_type6: [],
    });
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

  const statusOPTIONS = [
    { key: 1, value: '1' },
    { key: 2, value: '2' },
    { key: 3, value: '3' },
  ];
  const appsOPTIONS1 = [
    { key: 0, value: '特征名字' },
    { key: 1, value: '字符串' },
  ];
  const appsOPTIONS2 = [
    { key: 0, value: '特征名字' },
    { key: 1, value: '字符串' },
  ];
  const appsOPTIONS3 = [
    { key: 0, value: '特征名字' },
    { key: 1, value: '字符串' },
  ];
  const appsOPTIONS4 = [
    { key: 0, value: '特征名字' },
    { key: 1, value: '字符串' },
  ];
  const appsOPTIONS5 = [
    { key: 0, value: '特征名字' },
    { key: 1, value: '字符串' },
  ];
  const appsOPTIONS6 = [
    { key: 0, value: '特征名字' },
    { key: 1, value: '字符串' },
  ];
  const goToWhat = () => {
    let valueFromSelect: any = '';

    valueFromSelect =
      fromRowSelect.param_type1 +
      ',' +
      fromRowSelect.param_type2 +
      ',' +
      fromRowSelect.param_type3 +
      ',' +
      fromRowSelect.param_type4 +
      ',' +
      fromRowSelect.param_type5 +
      ',' +
      fromRowSelect.param_type6 
      // (fromRowSelect.param_type6.toString() !== '' ? ',' + fromRowSelect.param_type6 : '');

    const fromRowSelectSum = {
      ...fromRowSelect,
      param_type: valueFromSelect,
    };
    const featureRegisterOpPostQuestValue = Object.assign(fromRow, fromRowSelectSum);
    // console.log(valueFromSelect, featureRegisterOpPostQuestValue);

    api.featureRegisterOpPostQuest(featureRegisterOpPostQuestValue).then(item => {
      if (item.retcode === 0) {
        console.log(item, '999');

        message.success('查询成功');
      } else if (item.retmsg) {
        message.error(`失败原因: ${item.retmsg}`);
      }
    });
  };
  return (
    <div className="SceneRegistrationClass">
      <div className="bodyClass">
        <div className="SceneHeader">注册算子</div>
        <div className="site-card-border-less-wrapper">
          <Card title="基本信息" bordered={false}>
            <Row gutter={16}>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>算子英文名</span>
                  <Input
                    placeholder="请输入算子英文名"
                    style={{ width: 160 }}
                    value={fromRow.op_name}
                    onChange={fromFunChanger('op_name')}
                  />
                </div>
              </Col>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>算子中文名</span>
                  <Input
                    placeholder="请输入算子中文名"
                    style={{ width: 160 }}
                    value={fromRow.op_desc}
                    onChange={fromFunChanger('op_desc')}
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

          <Card title="参数信息" bordered={false}>
            <Row gutter={16}>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>参数数量</span>
                  <Select
                    placeholder="请选择参数数量"
                    value={fromRowSelect.param_count}
                    onChange={fromFunChangerTwo('param_count')}
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
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>参数1类型</span>
                  <Select
                    // mode="multiple"
                    allowClear
                    placeholder="请选择命名空间"
                    value={fromRowSelect.param_type1}
                    onChange={fromFunChangerTwo('param_type1')}
                    style={{ width: '200px' }}
                  >
                    {appsOPTIONS1.map(item => {
                      return (
                        <Select.Option key={item.key} value={item.key}>
                          {item.value}
                        </Select.Option>
                      );
                    })}
                  </Select>
                </div>
              </Col>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>参数2类型</span>
                  <Select
                    // mode="multiple"
                    allowClear
                    placeholder="请选择地区"
                    value={fromRowSelect.param_type2}
                    onChange={fromFunChangerTwo('param_type2')}
                    style={{ width: '200px' }}
                  >
                    {appsOPTIONS2.map(item => {
                      return (
                        <Select.Option key={item.key} value={item.key}>
                          {item.value}
                        </Select.Option>
                      );
                    })}
                  </Select>
                </div>
              </Col>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>参数3类型</span>
                  <Select
                    // mode="multiple"
                    allowClear
                    placeholder="请选择地区"
                    value={fromRowSelect.param_type3}
                    onChange={fromFunChangerTwo('param_type3')}
                    style={{ width: '200px' }}
                  >
                    {appsOPTIONS3.map(item => {
                      return (
                        <Select.Option key={item.key} value={item.key}>
                          {item.value}
                        </Select.Option>
                      );
                    })}
                  </Select>
                </div>
              </Col>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>参数4类型</span>
                  <Select
                    // mode="multiple"
                    allowClear
                    placeholder="请选择地区"
                    value={fromRowSelect.param_type4}
                    onChange={fromFunChangerTwo('param_type4')}
                    style={{ width: '200px' }}
                  >
                    {appsOPTIONS4.map(item => {
                      return (
                        <Select.Option key={item.key} value={item.key}>
                          {item.value}
                        </Select.Option>
                      );
                    })}
                  </Select>
                </div>
              </Col>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>参数5类型</span>
                  <Select
                    // mode="multiple"
                    allowClear
                    placeholder="请选择地区"
                    value={fromRowSelect.param_type5}
                    onChange={fromFunChangerTwo('param_type5')}
                    style={{ width: '200px' }}
                  >
                    {appsOPTIONS5.map(item => {
                      return (
                        <Select.Option key={item.key} value={item.key}>
                          {item.value}
                        </Select.Option>
                      );
                    })}
                  </Select>
                </div>
              </Col>
              <Col className="gutter-row" span={12}>
                <div style={style}>
                  <span style={{ width: '26%' }}>参数6类型</span>
                  <Select
                    // mode="multiple"
                    allowClear
                    placeholder="请选择地区"
                    value={fromRowSelect.param_type6}
                    onChange={fromFunChangerTwo('param_type6')}
                    style={{ width: '200px' }}
                  >
                    {appsOPTIONS6.map(item => {
                      return (
                        <Select.Option key={item.key} value={item.key}>
                          {item.value}
                        </Select.Option>
                      );
                    })}
                  </Select>
                </div>
              </Col>
            </Row>
          </Card>
          <Button type="primary" className="preservationClass" onClick={goToWhat}>
            保存
          </Button>
        </div>
      </div>
    </div>
  );
}
