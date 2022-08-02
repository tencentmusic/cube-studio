import React, { useState, useEffect } from 'react';
// import _ from 'lodash';
import { Card, Row, Col, Button, Input, Select, InputNumber, message, Form } from 'antd';
import { useHistory } from 'react-router-dom';
import { MinusCircleOutlined, PlusOutlined } from '@ant-design/icons';
import { FormInstance } from 'antd/es/form';

import api from '@src/api';
import './ReFeatureInformation.css';

const comStore = {
  uuid: Date.now(),
  getUUid() {
    return this.uuid++;
  },
};

export default function ReFeatureInformation() {
  const history = useHistory();
  const formRef = React.createRef<FormInstance>();
  const [fromRow, setFromRow] = useState({
      feature_set_id: '',
      feature_group_name: '',
      feature_group_desc: '',
      owner_rtxs: '',
      feature_user_slot: '',
      feature_item_slot: '',
      feature_name: '',
      feature_desc: '',
      unique_feature_name: '',
      timeout: '',
      hashId: '',
      resource_info: '',
      ReFeatureInformationTitle: '注册特征信息',
      group_id: '',
      sample_id: '',
      isomerismFeatureInfoArray: [{ feature_id: undefined }],
    }),
    [IsomerismFeatureInfoArray, setIsomerismFeatureInfoArray] = useState([
      {
        uuid: comStore.getUUid(),
        feature_id: '',
        feature_desc: '',
        feature_item_slot: undefined,
        feature_name: '',
        feature_user_slot: '',
        feature_type: [],
        numer_type: [],
        ReFeatureInformationTitle: '注册特征信息',
      },
    ]),
    [FEATUREValue, setFEATUREValue] = useState([{ label: '', value: '' }]),
    [FEATURETYPEValue, setFEATURETYPEValue] = useState([{ label: '', value: '' }]),
    [FEATURESTYLEValue, setFEATURESTYLEValue] = useState([{ label: '', value: '' }]),
    [featureServRouter, setFeatureServRouter] = useState([
      { feature_serv_name: '', feature_serv_id: '', feature_serv_type: '' },
    ]),
    [featureServRouterArray, setfeatureServRouterArray] = useState([
      { feature_serv_name: '', feature_serv_id: '', feature_serv_type: '' },
    ]),
    [RemoveName, setRemoveName] = useState([]),
    [hashIdDisabled, setHashIdDisabled] = useState(true),
    [DATASOURCESValue, setDATASOURCESValue] = useState([{ label: '', value: '' }]);

  function featureDisplaySetGetQuestHandle(feature_set_idID: any) {
    api.featureDisplaySetGetQuest(feature_set_idID, true, true).then(item => {
      if (item.retcode === 0) {
        const featureServRouter = item.result.featureServRouter;

        setFeatureServRouter(featureServRouter);
      } else if (item.retmsg) {
        message.error(`失败原因: ${item.retmsg}`);
      }
    });
  }
  useEffect(() => {
    const valueSum = { class: 'FEATURE,DATASOURCES,FEATURETYPE,FEATURESTYLE' };

    api.featureKVDataDisplayPostQuest(valueSum).then(item => {
      if (item.retcode === 0) {
        const FEATURE = item.result.data.FEATURE;
        const DATASOURCES = item.result.data.DATASOURCES;
        const FEATURETYPE = item.result.data.FEATURETYPE;
        const FEATURESTYLE = item.result.data.FEATURESTYLE;
        setFEATUREValue(FEATURE);
        setDATASOURCESValue(DATASOURCES);
        setFEATURETYPEValue(FEATURETYPE);
        setFEATURESTYLEValue(FEATURESTYLE);
        message.success('查询成功');
      } else if (item.retmsg) {
        message.error(`失败原因: ${item.retmsg}`);
      }
    });

    let setFromRow_value: any = {};

    if (history.location.state) {
      //  判断当前有参数
      setFromRow_value = history.location.state;
      sessionStorage.setItem('keyIDChilder', JSON.stringify(setFromRow_value)); // 存入到sessionStorage中
      if (setFromRow_value.source_type === '4') {
        setHashIdDisabled(false);
      } else {
        setHashIdDisabled(true);
      }
      setFromRow({ ...setFromRow_value });

      if (setFromRow_value.ReFeatureInformationTitle === '修改注册特征信息') {
        featureDisplaySetGetQuestHandle(setFromRow_value.feature_set_id);
        formRef.current?.setFieldsValue({
          ...setFromRow_value,
          strategy_ids: setFromRow_value.strategy_ids,
        });
        // let j = 0;
        // const timer = setInterval(function () {
        //   j += 50;
        //   setIsomerismFeatureInfoArray(setFromRow_value.isomerismFeatureInfoArray.slice(0, j));
        //   if (j > setFromRow_value.isomerismFeatureInfoArray.length) {
        //     clearInterval(timer);
        //   }
        // }, 1000);
        setIsomerismFeatureInfoArray(setFromRow_value.isomerismFeatureInfoArray);
      } else {
        formRef.current?.setFieldsValue({
          ...setFromRow_value,
          strategy_ids: setFromRow_value.strategy_ids,
        });

        setIsomerismFeatureInfoArray(
          setFromRow_value.isomerismFeatureInfoArray ? setFromRow_value.isomerismFeatureInfoArray : [],
        );
        featureDisplaySetGetQuestHandle(setFromRow_value.sample_id);
      }
    } else {
      setFromRow_value = JSON.parse(sessionStorage.getItem('keyIDChilder') || ''); // 当state没有参数时，取sessionStorage中的参数
      if (setFromRow_value.source_type === '4') {
        setHashIdDisabled(false);
      } else {
        setHashIdDisabled(true);
      }

      setFromRow({ ...setFromRow_value });

      if (setFromRow_value.ReFeatureInformationTitle === '修改注册特征信息') {
        featureDisplaySetGetQuestHandle(setFromRow_value.feature_set_id);
        formRef.current?.setFieldsValue({
          ...setFromRow_value,
          strategy_ids: setFromRow_value.strategy_ids,
        });

        // let i = 0;
        // const timer = setInterval(function () {
        //   i += 50;
        //   setIsomerismFeatureInfoArray(setFromRow_value.isomerismFeatureInfoArray.slice(0, i));
        //   if (i > setFromRow_value.isomerismFeatureInfoArray.length) {
        //     clearInterval(timer);
        //   }
        // }, 1000);
        setIsomerismFeatureInfoArray(setFromRow_value.isomerismFeatureInfoArray);
      } else {
        formRef.current?.setFieldsValue({
          ...setFromRow_value,
          strategy_ids: setFromRow_value.strategy_ids,
        });

        setIsomerismFeatureInfoArray(
          setFromRow_value.isomerismFeatureInfoArray ? setFromRow_value.isomerismFeatureInfoArray : [],
        );
        featureDisplaySetGetQuestHandle(setFromRow_value.sample_id || setFromRow_value.feature_set_id);
      }
    }
  }, []);
  useEffect(() => {
    for (let i = 0; i < IsomerismFeatureInfoArray.length; i++) {
      formRef.current?.setFieldsValue({
        [`feature_id${i}`]: IsomerismFeatureInfoArray[i].feature_id,
        [`feature_name${i}`]: IsomerismFeatureInfoArray[i].feature_name,
        [`feature_desc${i}`]: IsomerismFeatureInfoArray[i].feature_desc,
        [`feature_type${i}`]: IsomerismFeatureInfoArray[i].feature_type,
        [`numer_type${i}`]: IsomerismFeatureInfoArray[i].numer_type,
        [`feature_user_slot${i}`]: IsomerismFeatureInfoArray[i].feature_user_slot,
        [`feature_item_slot${i}`]: IsomerismFeatureInfoArray[i].feature_item_slot,
        [`hashId${i}`]: (IsomerismFeatureInfoArray[i] as any).hashId,
        [`unique_feature_name${i}`]: (IsomerismFeatureInfoArray[i] as any).unique_feature_name,
      });
    }
  }, [IsomerismFeatureInfoArray]);

  const addMeta = async () => {
    try {
      const costInfoArrCopy: any = await beforeSetDataSync();

      setIsomerismFeatureInfoArray([
        ...costInfoArrCopy,
        {
          uuid: comStore.getUUid(),
          feature_id: '',
          feature_desc: costInfoArrCopy.feature_desc,
          feature_item_slot: costInfoArrCopy.feature_item_slot,
          feature_name: '',
          feature_user_slot: '',
          feature_type: [],
          numer_type: [],
          ReFeatureInformationTitle: '注册特征信息',
        },
      ]);
    } catch (err) {
      console.error(err);
    }
  };
  const beforeSetDataSync = () => {
    return new Promise((suc, fail) => {
      try {
        const values = formRef.current?.getFieldsValue();

        const numEnd = /(\d+)$/;
        // 所有以数字结尾的key
        const numObj: any = [];
        // 其余的key
        const resetObj: any = {};
        Object.entries(values).forEach(([key, value]) => {
          if (numEnd.test(key)) {
            const num = RegExp.$1;
            if (!numObj[num]) {
              numObj[num] = {};
            }
            numObj[num][key.substring(0, key.length - String(num).length)] = value;
          } else {
            resetObj[key] = value;
          }
        });
        const numObj_value = numObj.map((item: any) => {
          return { ...item, uuid: comStore.getUUid() };
        });

        suc(numObj_value);
      } catch (err) {
        fail(err);
      }
    });
  };

  const remove = async (name: any) => {
    const removeName: any = RemoveName.slice();
    removeName.push(name);
    setRemoveName(removeName);

    if (name.uuid) {
      setIsomerismFeatureInfoArray(IsomerismFeatureInfoArray.filter((item: any) => item.uuid !== name.uuid));
    }
    if (!name.uuid) {
      setIsomerismFeatureInfoArray(
        IsomerismFeatureInfoArray.filter(item => (item.feature_id as any) !== name.feature_id),
      );
    }
  };

  const onFinish = (values: any) => {
    const numEnd = /(\d+)$/;
    // 所有以数字结尾的key
    const numObj: any = [];
    // 其余的key
    const resetObj: any = {};
    Object.entries(values).forEach(([key, value]) => {
      if (numEnd.test(key)) {
        const num = RegExp.$1;
        if (!numObj[num]) {
          numObj[num] = {};
        }
        numObj[num][key.substring(0, key.length - String(num).length)] = value;
      } else {
        resetObj[key] = value;
      }
    });

    const _feature_id = IsomerismFeatureInfoArray[IsomerismFeatureInfoArray.length - 1].feature_id;

    if (fromRow.ReFeatureInformationTitle === '查看注册特征信息') {
      history.push('/HeterogeneousPlatform/Nationalkaraoke/FeatureSetConfiguration');
    }

    if (fromRow.ReFeatureInformationTitle === '修改注册特征信息') {
      if (fromRow.group_id) {
        const _RemoveName = RemoveName.map((item: any) => {
          return {
            feature_id: item.feature_id,
            isDel: 1,
          };
        });

        const numObj_value = numObj.concat(_RemoveName);

        const valueSum = Object.assign(
          {
            ...resetObj,
            group_id: fromRow.group_id,
          },
          {
            isomerismFeatureInfoArray: [...numObj_value],
          },
        );

        const object_value: any = {};
        const getValueByKey = (fromRow: any, valueSum: any) => {
          if (
            Object.prototype.toString.call(fromRow) === '[object Object]' &&
            Object.prototype.toString.call(valueSum) === '[object Object]'
          ) {
            for (const key in fromRow) {
              if (Object.prototype.hasOwnProperty.call(fromRow, key)) {
                const element = fromRow[key];
                const source = valueSum[key];
                if (Array.isArray(element)) {
                  for (let i = 0; i < element.length; i++) {
                    for (let j = 0; j < source.length; j++) {
                      getValueByKey(element[i], source[j]);
                    }
                  }
                } else {
                  if (fromRow[key] !== valueSum[key]) {
                    object_value[key] = valueSum[key];
                  }
                }
              }
            }
          }
        };

        getValueByKey(fromRow, valueSum);

        api.featureUpdateIsomerismFeatureFeatureInfoQUEST(valueSum).then(item => {
          if (item.retcode === 0) {
            history.push('/HeterogeneousPlatform/Nationalkaraoke/FeatureSetConfiguration');
            message.success('修改注册特征信息');
          } else if (item.retmsg) {
            message.error(`失败原因: ${item.retmsg}`);
          }
        });

        if (_feature_id === '') {
          const _numObj = numObj.filter((item: any) => {
            if (item.feature_id) {
              return false;
            }
            delete item.feature_id;
            return item;
          });

          const valueSum = Object.assign(
            {
              group_id: fromRow.group_id,
            },
            {
              isomerismFeatureInfoArray: [..._numObj],
            },
          );
          api.featureRegisterIsomerismFeatureFeatureInfoPostQuest(valueSum).then(item => {
            if (item.retcode === 0) {
              history.push('/HeterogeneousPlatform/Nationalkaraoke/FeatureSetConfiguration');

              message.success('新增成功');
            } else if (item.retmsg) {
              message.error(`失败原因: ${item.retmsg}`);
            }
          });
        }
      }
    } else if (fromRow.ReFeatureInformationTitle === '注册特征信息') {
      const _numObj = numObj.filter((item: any) => {
        if (item.feature_id) {
          return false;
        }
        delete item.feature_id;
        return item;
      });
      const valueSum = Object.assign(
        {
          ...resetObj,
          feature_set_id: fromRow.sample_id || fromRow.feature_set_id,
        },
        {
          isomerismFeatureInfoArray: [..._numObj],
        },
      );
      api.featureRegisterIsomerismFeatureFeatureInfoPostQuest(valueSum).then(item => {
        if (item.retcode === 0) {
          history.push('/HeterogeneousPlatform/Nationalkaraoke/FeatureSetConfiguration');

          message.success('查询成功');
        } else if (item.retmsg) {
          message.error(`失败原因: ${item.retmsg}`);
        }
      });
    }
  };
  const DATASOURCESValueHandle = (value: string) => {
    if (value === '4') {
      setHashIdDisabled(false);
    } else {
      setHashIdDisabled(true);
    }
    formRef.current?.setFieldsValue({
      resource_info: '',
    });

    const featureServRouterArray = featureServRouter.filter((item: any) => {
      if (item.feature_serv_type === value) {
        return { value: item.feature_serv_id, key: item.feature_serv_id, label: item.feature_serv_name };
      }
    });

    setfeatureServRouterArray(featureServRouterArray);
  };
  const config = {
    rules: [{ required: true, message: 'Please select time!' }],
  };
  return (
    <div className="SceneRegistrationClass">
      <div className="bodyClass">
        <div className="SceneHeader">{fromRow.ReFeatureInformationTitle}</div>
        <div className="site-card-border-less-wrapper">
          <Form ref={formRef} name="IsomerismFeatureInfoArray" onFinish={onFinish} autoComplete="off" layout="vertical">
            <Card title="基本信息" bordered={false}>
              <Row gutter={16}>
                <Col className="gutter-row" span={8}>
                  <Form.Item label="特征组id" name={`feature_set_id`}>
                    <InputNumber disabled placeholder="请输入物品slot" style={{ width: 200 }} />
                  </Form.Item>
                </Col>
                <Col className="gutter-row" span={8}>
                  <Form.Item label="特征组类型" name={`feature_group_type`} {...config}>
                    <Select placeholder="请选择特征组类型" style={{ width: '200px' }}>
                      {FEATUREValue.map(item => {
                        return (
                          <Select.Option key={item.label} value={item.value}>
                            {item.label}
                          </Select.Option>
                        );
                      })}
                    </Select>
                  </Form.Item>
                </Col>
                <Col className="gutter-row" span={8}>
                  <Form.Item label="特征组名称" name={`feature_group_name`} {...config}>
                    <Input placeholder="请输入特征组名称" style={{ width: 200 }} />
                  </Form.Item>
                </Col>
                <Col className="gutter-row" span={8}>
                  <Form.Item label="特征组描述" name={`feature_group_desc`} {...config}>
                    <Input placeholder="请输入特征组描述" style={{ width: 200 }} />
                  </Form.Item>
                </Col>

                <Col className="gutter-row" span={8}>
                  <Form.Item label="责任人" name={`owner_rtxs`} {...config}>
                    <Input placeholder="请输入责任人" style={{ width: 200 }} />
                  </Form.Item>
                </Col>
              </Row>
            </Card>

            <Card title="存储信息" bordered={false}>
              <Row gutter={16}>
                <Col className="gutter-row" span={8}>
                  <Form.Item label="数据来源" name={`source_type`} {...config}>
                    <Select placeholder="请选择数据来源" style={{ width: '200px' }} onChange={DATASOURCESValueHandle}>
                      {DATASOURCESValue.map(item => {
                        return (
                          <Select.Option key={item.label} value={item.value}>
                            {item.label}
                          </Select.Option>
                        );
                      })}
                    </Select>
                  </Form.Item>
                </Col>

                <Col className="gutter-row" span={8}>
                  <Form.Item label="特征服务" name={`resource_info`} {...config}>
                    <Select placeholder="请选择特征服务" style={{ width: '200px' }}>
                      {featureServRouterArray.map(item => {
                        return (
                          <Select.Option key={item.feature_serv_id} value={item.feature_serv_id}>
                            {item.feature_serv_name}
                          </Select.Option>
                        );
                      })}
                    </Select>
                  </Form.Item>
                </Col>

                <Col className="gutter-row" span={8}>
                  <Form.Item label="过期时间" name={`timeout`} {...config}>
                    <InputNumber addonAfter="ms" placeholder="请输入过期时间" style={{ width: 200 }} />
                  </Form.Item>
                </Col>
              </Row>
            </Card>

            <Card title="特征字段信息" bordered={false}>
              <Form.Item>
                {IsomerismFeatureInfoArray.map((item: any, index) => {
                  return (
                    <Row gutter={16} key={index}>
                      <Col className="gutter-row" span={8}>
                        <Form.Item label="特征ID" name={`feature_id${index}`}>
                          <Input disabled placeholder="请输入特征英文名" style={{ width: 200 }} />
                        </Form.Item>
                      </Col>
                      <Col className="gutter-row" span={8}>
                        <Form.Item label="特征英文名" name={`feature_name${index}`} {...config}>
                          <Input
                            disabled={item.feature_name ? true : false}
                            placeholder="请输入特征英文名"
                            style={{ width: 200 }}
                          />
                        </Form.Item>
                      </Col>
                      <Col className="gutter-row" span={8}>
                        <Form.Item label="特征中文名" name={`feature_desc${index}`} {...config}>
                          <Input placeholder="请输入特征中文名" style={{ width: 200 }} />
                        </Form.Item>
                      </Col>
                      <Col className="gutter-row" span={8}>
                        <Form.Item label="唯一特征名" name={`unique_feature_name${index}`}>
                          <Input disabled placeholder="请输入唯一特征名" style={{ width: 200 }} />
                        </Form.Item>
                      </Col>
                      <Col className="gutter-row" span={8}>
                        <Form.Item label="特征格式" name={`feature_type${index}`} {...config}>
                          <Select placeholder="请选择特征服务" style={{ width: '200px' }}>
                            {FEATURESTYLEValue.map(item => {
                              return (
                                <Select.Option key={item.label} value={item.value}>
                                  {item.label}
                                </Select.Option>
                              );
                            })}
                          </Select>
                        </Form.Item>
                      </Col>
                      <Col className="gutter-row" span={8}>
                        <Form.Item label="特征类型" name={`numer_type${index}`} {...config}>
                          <Select placeholder="请选择特征类型" style={{ width: '200px' }}>
                            {FEATURETYPEValue.map(item => {
                              return (
                                <Select.Option key={item.label} value={item.value}>
                                  {item.label}
                                </Select.Option>
                              );
                            })}
                          </Select>
                        </Form.Item>
                      </Col>
                      <Col className="gutter-row" span={8}>
                        <Form.Item label="用户slot" name={`feature_user_slot${index}`} {...config}>
                          <InputNumber placeholder="请输入用户slot" style={{ width: 200 }} />
                        </Form.Item>
                      </Col>
                      <Col className="gutter-row" span={8}>
                        <Form.Item label="物品slot" name={`feature_item_slot${index}`} {...config}>
                          <InputNumber placeholder="请输入物品slot" style={{ width: 200 }} />
                        </Form.Item>
                      </Col>
                      <Col className="gutter-row" span={8}>
                        <Form.Item label="hashId" name={`hashId${index}`}>
                          <Input disabled={hashIdDisabled} style={{ width: 200 }} />
                        </Form.Item>
                      </Col>
                      <Col className="gutter-row" span={8}>
                        <Form.Item>
                          <Button
                            className="ButtonClass"
                            type="dashed"
                            onClick={() => remove(item)}
                            block
                            icon={<MinusCircleOutlined />}
                          ></Button>
                        </Form.Item>
                      </Col>
                    </Row>
                  );
                })}
              </Form.Item>
              <Form.Item>
                <Button type="dashed" onClick={addMeta} block icon={<PlusOutlined />}>
                  新增
                </Button>
              </Form.Item>
              <Form.Item>
                <div className="preservationClass">
                  <Button
                    style={{
                      backgroundColor: 'rgba(255, 87, 51, 1)',
                      width: '140px',
                      height: '40px',
                      marginRight: '10px',
                    }}
                    onClick={() => history.go(-1)}
                    type="primary"
                  >
                    返回
                  </Button>
                  <Button
                    type="primary"
                    htmlType="submit"
                    style={{ backgroundColor: 'rgba(255, 87, 51, 1)', width: ' 140px', height: '40px' }}
                  >
                    {fromRow.ReFeatureInformationTitle === '查看注册特征信息' ? '返回' : '保存'}
                  </Button>
                </div>
              </Form.Item>
            </Card>
          </Form>
        </div>
      </div>
    </div>
  );
}
