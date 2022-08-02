import React, { useState, useEffect } from 'react';
import { Modal, Form, Input, Select, message, InputNumber } from 'antd';
import api from '@src/api';
interface Values {
  title: string;
  description: string;
  modifier: string;
}

interface CollectionCreateFormProps {
  visible: boolean;
  onCreate: (values: Values) => void;
  onCancel: () => void;
  VisibleSee_title: any;
}

const CollectionCreateForm: React.FC<CollectionCreateFormProps> = ({
  visible,
  onCreate,
  onCancel,
  VisibleSee_title,
}) => {
  const [form] = Form.useForm();
  const [DATASOURCESValue, setDATASOURCESValue] = useState([{ label: '', value: '' }]),
    [ROUTERSELECT, setROUTERSELECT] = useState([{ label: '', value: '' }]),
    [NAMESAPCE, setNAMESAPCE] = useState([{ label: '', value: '' }]),
    [REGION, setREGION] = useState([{ label: '', value: '' }]);
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
  }, []);
  return (
    <Modal
      // footer={null}
      visible={visible}
      title={VisibleSee_title.title}
      okText="确认"
      cancelText="取消"
      onCancel={onCancel}
      onOk={() => {
        form
          .validateFields()
          .then(values => {
            form.resetFields();
            onCreate(values);
          })
          .catch(info => {
            console.log('Validate Failed:', info);
          });
      }}
    >
      <Form form={form} name="form_in_modal" initialValues={VisibleSee_title.record}>
        <Form.Item name="bid" label="bid">
          <InputNumber
            disabled={VisibleSee_title.title === '查看特征配置' ? true : false}
            placeholder="请输入bid"
            style={{ width: 200 }}
          />
        </Form.Item>
        <Form.Item name="cre_time" label="创建时间">
          <Input disabled={VisibleSee_title.title === '查看特征配置' ? true : false} />
        </Form.Item>
        <Form.Item name="feature_serv_desc" label="服务中文名">
          <Input disabled={VisibleSee_title.title === '查看特征配置' ? true : false} />
        </Form.Item>
        <Form.Item name="feature_serv_name" label="服务英文名">
          <Input disabled={VisibleSee_title.title === '查看特征配置' ? true : false} />
        </Form.Item>
        <Form.Item name="feature_serv_id" label="ID">
          <Input disabled={VisibleSee_title.title === '查看特征配置' ? true : false} />
        </Form.Item>

        <Form.Item name="target" label="服务名字">
          <Input disabled={VisibleSee_title.title === '查看特征配置' ? true : false} />
        </Form.Item>

        <Form.Item name="feature_serv_type" label="服务来源类型">
          <Select
            disabled={VisibleSee_title.title === '查看特征配置' ? true : false}
            placeholder="请选择数据来源"
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
        </Form.Item>
        <Form.Item name="selector_name" label="路由选择器">
          <Select
            disabled={VisibleSee_title.title === '查看特征配置' ? true : false}
            placeholder="请选择路由选择器"
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
        </Form.Item>

        <Form.Item name="namesapce" label="命名空间">
          <Select
            disabled={VisibleSee_title.title === '查看特征配置' ? true : false}
            placeholder="请选择命名空间"
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
        </Form.Item>
        <Form.Item name="region" label="地区">
          <Select
            disabled={VisibleSee_title.title === '查看特征配置' ? true : false}
            placeholder="请选择地区"
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
        </Form.Item>

        <Form.Item name="last_mod" label="上次修改时间">
          <Input disabled={VisibleSee_title.title === '查看特征配置' ? true : false} />
        </Form.Item>

        <Form.Item name="owner_rtxs" label="责任人">
          <Input disabled={VisibleSee_title.title === '查看特征配置' ? true : false} />
        </Form.Item>
      </Form>
    </Modal>
  );
};
export default CollectionCreateForm;
