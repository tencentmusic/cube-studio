import React, { useState } from 'react';


import { Modal, Form, Input, Button } from 'antd';
import ChildrenTable from './ChildrenTable';
import api from '@src/api';

// interface Values {
//   opr: string;
//   ips: string;
// }

interface CollectionCreateFormProps {
  visible: boolean;
  // onCreate: (values: Values) => void;
  onCancel: () => void;
}
// 查看组件信息

const ConfigurationValidation: React.FC<CollectionCreateFormProps> = ({ visible, onCancel }) => {
  const [form] = Form.useForm();

  // function handleChangeTtypeSelect(value: any) {
  //   console.log(`selected ${value}`);
  // }
  const onFinish = (values: any) => {
    api.config_check(values).then(item => {
      setDataValue(item.result.data || []);
    });
    // form.resetFields();
  };
  const columns = [
    {
      title: '组件',
      dataIndex: 'mark',
     
    },
    {
      title: '配置中心',
      dataIndex: 'config_center',
      /* eslint-disable */
      render: (text: any, record: any) => {
        return <div style={{ color: record.config_center === 'NO' ? 'red' : '' }}>{record.config_center}</div>;
      },
    },
    {
      title: '召回服务',
      dataIndex: 'server',
      /* eslint-disable */
      render: (text: any, record: any) => (
        <div style={{ color: record.server === 'NO' ? 'red' : '' }}>{record.server}</div>
      ),
    },
  ];

  const [columnsValue, setcolumnsValue] = useState(columns),
    [dataValue, setDataValue] = useState([]);

  return (
    <Modal width={1000} footer={null} visible={visible} title="配置效验" onCancel={onCancel}>
      <Form style={{ marginBottom: '20px' }} form={form} layout="inline" name="form_in_modal" onFinish={onFinish}>
        <Form.Item
          name="ips"
          label="address"
          rules={[{ required: true, message: 'Please input the opr of collection!' }]}
        >
          <Input style={{ width: 200 }} />
        </Form.Item>

        <Form.Item wrapperCol={{ offset: 8, span: 16 }}>
          <Button type="primary" htmlType="submit">
            效验
          </Button>
        </Form.Item>
      </Form>
      <ChildrenTable columns={columnsValue} data={dataValue} />
    </Modal>
  );
};
export default ConfigurationValidation;
