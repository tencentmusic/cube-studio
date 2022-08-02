import React, { useState } from 'react';
import { Modal, Form, Input } from 'antd';
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
          <Input />
        </Form.Item>
        <Form.Item name="cre_time" label="创建时间">
          <Input />
        </Form.Item>
        <Form.Item name="feature_serv_desc" label="服务中文名">
          <Input />
        </Form.Item>
        <Form.Item name="feature_serv_id" label="ID">
          <Input />
        </Form.Item>

        <Form.Item name="feature_serv_name" label="服务名字">
          <Input />
        </Form.Item>

        <Form.Item name="feature_serv_type" label="服务来源类型">
          <Input />
        </Form.Item>

        <Form.Item name="namesapce" label="命名空间">
          <Input />
        </Form.Item>
        <Form.Item name="region" label="地区">
          <Input />
        </Form.Item>

        <Form.Item name="last_mod" label="上次修改时间">
          <Input />
        </Form.Item>

        <Form.Item name="owner_rtxs" label="责任人">
          <Input />
        </Form.Item>
      </Form>
    </Modal>
  );
};
export default CollectionCreateForm;
