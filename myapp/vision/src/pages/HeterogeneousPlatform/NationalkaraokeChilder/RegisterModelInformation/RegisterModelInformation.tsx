import React, { useState, useEffect } from 'react';
import { Form, Button, Input, Select, Row, Col, message } from 'antd';
import { PlusOutlined, MinusCircleOutlined } from '@ant-design/icons';
import { FormInstance } from 'antd/es/form';
import api from '@src/api';
import { useHistory } from 'react-router-dom';
import './RegisterModelInformation.css';
const comStore = {
  uuid: Date.now(),
  getUUid() {
    return this.uuid++;
  },
};
export default function RegisterModelInformation() {
  const history = useHistory();
  const formRef = React.createRef<FormInstance>();
  const [dynamicInput, setDynamicInput] = useState([{ preds: '', uuid: comStore.getUUid() }]);
  const [fe_proc_idVlaue, setfe_proc_idVlaue] = useState([{ fe_proc_id: '', fe_proc_name: '' }]);
  const [model_serv_idVlaue, setModel_serv_idVlaue] = useState([{ model_serv_id: '', model_serv_name: '' }]);
  const [scene_id_ID, setscene_id_ID] = useState({
    scene_id: '',
    preds: '',
    title: '',
    model_id: '',
    fe_proc_id: { fe_proc_id: '' },
    resource_info: { model_serv_id: '' },
  });

  const onFinish = (values: any) => {
    const params: any = {};
    const predsArr = [] as any;
    for (const key in values) {
      if (key.includes('preds')) {
        predsArr.push(values[key]);
      } else {
        params[key] = values[key];
      }
    }
    params.preds = predsArr.toString();
    if (scene_id_ID.title === '更改注册模型信息') {
      api
        .featureUpdateModelConfigPostQuest({
          ...params,
          scene_id: scene_id_ID.scene_id,
          fe_proc_id: scene_id_ID.fe_proc_id.fe_proc_id,

          model_id: scene_id_ID.model_id,
        })
        .then((item: any) => {
          if (item.retcode === 0) {
            history.push('/HeterogeneousPlatform/Nationalkaraoke/SceneModelInformation');
          } else if (item.retmsg) {
            message.error(`失败原因: ${item.retmsg}`);
          }
        });
    } else {
      api.featureRegisterModelConfigPostQuest({ ...params, scene_id: scene_id_ID.scene_id }).then((item: any) => {
        if (item.retcode === 0) {
          history.push('/HeterogeneousPlatform/Nationalkaraoke/SceneModelInformation');
        } else if (item.retmsg) {
          message.error(`失败原因: ${item.retmsg}`);
        }
      });
    }
  };
  useEffect(() => {
    let setFromRow_value: any = {};

    if (history.location.state) {
      //  判断当前有参数
      setFromRow_value = history.location.state;

      sessionStorage.setItem('registerModelInformationKeyID', JSON.stringify(setFromRow_value)); // 存入到sessionStorage中
      setscene_id_ID(setFromRow_value);
      //  model_serv_router
      api.featureFeProcConfigDisplayGetQuest(setFromRow_value.scene_id, 'fe_proc_config').then(item => {
        setfe_proc_idVlaue(item.result);
      });
      api.featureFeProcConfigDisplayGetQuest(setFromRow_value.scene_id, 'model_serv_router').then(item => {
        setModel_serv_idVlaue(item.result);
      });

      if (setFromRow_value.title === '更改注册模型信息') {
        formRef.current?.setFieldsValue({
          model_name: setFromRow_value.model_name,
          model_desc: setFromRow_value.model_desc,
          owner_rtxs: setFromRow_value.owner_rtxs,
          resource_info: setFromRow_value.resource_info.model_serv_id,
          fe_proc_id: setFromRow_value.fe_proc_id.fe_proc_name,
        });
        let preds_value: any = {};

        preds_value = setFromRow_value.preds.split(',').map((item: any, index: any) => {
          return { preds: item };
        });

        setDynamicInput(setFromRow_value.preds.split(',') ? preds_value : []);
      }
    } else {
      setFromRow_value = JSON.parse(sessionStorage.getItem('registerModelInformationKeyID') || ''); // 当state没有参数时，取sessionStorage中的参数
      setscene_id_ID(setFromRow_value);

      api.featureFeProcConfigDisplayGetQuest(setFromRow_value.scene_id, 'fe_proc_config').then(item => {
        setfe_proc_idVlaue(item.result);
      });
      api.featureFeProcConfigDisplayGetQuest(setFromRow_value.scene_id, 'model_serv_router').then(item => {
        setModel_serv_idVlaue(item.result);
      });

      if (setFromRow_value.title === '更改注册模型信息') {
        formRef.current?.setFieldsValue({
          model_name: setFromRow_value.model_name,
          model_desc: setFromRow_value.model_desc,
          owner_rtxs: setFromRow_value.owner_rtxs,
          resource_info: setFromRow_value.resource_info.model_serv_id,
          fe_proc_id: setFromRow_value.fe_proc_id.fe_proc_name,
        });
        let preds_value: any = {};

        preds_value = setFromRow_value.preds.split(',').map((item: any, index: any) => {
          return { preds: item };
        });

        setDynamicInput(setFromRow_value.preds.split(',') ? preds_value : []);
      }
    }
  }, []);
  useEffect(() => {
    for (let i = 0; i < dynamicInput.length; i++) {
      formRef.current?.setFieldsValue({
        [`preds${i}`]: dynamicInput[i].preds,
      });
    }
  }, [dynamicInput]);
  const onFinishFailed = (errorInfo: any) => {
    console.log('Failed:', errorInfo);
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
            numObj[num][key.substring(0, key.length - 1)] = value;
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
  //  新增
  const addDynamicInput = async () => {
    try {
      const costInfoArrCopy: any = await beforeSetDataSync();

      setDynamicInput([...costInfoArrCopy, { preds: '', uuid: comStore.getUUid() }]);
    } catch (err) {
      console.error(err);
    }
  };
  const remove = (name: any) => {
    // const removeName: any = RemoveName.slice();
    // removeName.push(name);
    // setRemoveName(removeName);

    if (name.uuid) {
      setDynamicInput(dynamicInput.filter(item => item.uuid !== name.uuid));
    }
    if (!name.uuid) {
      setDynamicInput(dynamicInput.filter(item => item !== name));
    }
  };

  const goToRegisterModelService = () => {
    history.push('/HeterogeneousPlatform/Nationalkaraoke/RegisterModelService');
  };
  const goToFeatureConfiguration = () => {
    history.push({
      pathname: '/HeterogeneousPlatform/Nationalkaraoke/FeatureConfiguration',
      state: { ...scene_id_ID, title: '特征插件配置' },
    });
  };

  return (
    <div className="RegisterModelInformationClass">
      <div className="bodyClass">
        <div className="SceneHeader">注册模型信息</div>
        <Form
          name="basic"
          labelCol={{ span: 8 }}
          wrapperCol={{ span: 16 }}
          // initialValues={{ remember: true }}
          onFinish={onFinish}
          onFinishFailed={onFinishFailed}
          autoComplete="off"
          ref={formRef}
        >
          <Form.Item label="模型中文名" name="model_name" rules={[{ required: true, message: '请输入模型中文名' }]}>
            <Input style={{ width: 160 }} />
          </Form.Item>

          <Form.Item label="模型英文名" name="model_desc" rules={[{ required: true, message: '请输入模型英文名' }]}>
            <Input style={{ width: 160 }} />
          </Form.Item>

          <Form.Item label="预测目标">
            {dynamicInput.map((item, index) => {
              return (
                <Row gutter={16} key={item.uuid}>
                  <Col className="gutter-row" span={16}>
                    <Form.Item name={`preds${index}`} rules={[{ required: true, message: '请输入' }]}>
                      <Input style={{ width: 160 }} />
                    </Form.Item>
                  </Col>
                  <Col className="gutter-row" span={4}>
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
                  <Col className="gutter-row" span={4}>
                    <Form.Item>
                      <Button
                        className="ButtonClass"
                        type="dashed"
                        onClick={() => addDynamicInput()}
                        block
                        icon={<PlusOutlined />}
                      ></Button>
                    </Form.Item>
                  </Col>
                </Row>
              );
            })}
          </Form.Item>

          <Form.Item hasFeedback label="特征插件配置">
            <Row gutter={10}>
              <Col className="gutter-row" span={16}>
                <Form.Item name="fe_proc_id" rules={[{ required: true, message: '请输入特征插件配置' }]}>
                  <Select placeholder="Please select a country" style={{ width: 160 }}>
                    {fe_proc_idVlaue.map(item => {
                      return (
                        <Select.Option key={item.fe_proc_id} value={item.fe_proc_id}>
                          {item.fe_proc_name}
                        </Select.Option>
                      );
                    })}
                  </Select>
                </Form.Item>
              </Col>

              <Col className="gutter-row" span={4}>
                <Form.Item>
                  <Button
                    onClick={goToFeatureConfiguration}
                    className="ButtonClass"
                    type="dashed"
                    block
                    icon={<PlusOutlined />}
                  ></Button>
                </Form.Item>
              </Col>
            </Row>
          </Form.Item>

          <Form.Item label="模型服务信息" hasFeedback>
            <Row gutter={16}>
              <Col className="gutter-row" span={16}>
                <Form.Item name="resource_info" rules={[{ required: true, message: '请输入模型服务信息' }]}>
                  <Select placeholder="Please select a country" style={{ width: 160 }}>
                    {model_serv_idVlaue.map(item => {
                      return (
                        <Select.Option key={item.model_serv_id} value={item.model_serv_id}>
                          {item.model_serv_name}
                        </Select.Option>
                      );
                    })}
                  </Select>
                </Form.Item>
              </Col>

              <Col className="gutter-row" span={4}>
                <Form.Item>
                  <Button
                    onClick={goToRegisterModelService}
                    className="ButtonClass"
                    type="dashed"
                    block
                    icon={<PlusOutlined />}
                  ></Button>
                </Form.Item>
              </Col>
            </Row>
          </Form.Item>

          <Form.Item label="责任人" name="owner_rtxs" rules={[{ required: true, message: '请输入责任人' }]}>
            <Input style={{ width: 160 }} />
          </Form.Item>
          <Form.Item wrapperCol={{ offset: 8, span: 16 }}>
            <Button className="preservationClass" type="primary" htmlType="submit">
              保存
            </Button>
          </Form.Item>
        </Form>
      </div>
    </div>
  );
}
