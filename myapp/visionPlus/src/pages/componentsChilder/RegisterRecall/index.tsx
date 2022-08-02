import React from 'react';
import { Modal, Form, Input, Tooltip, Row, Col } from 'antd';
import { QuestionCircleOutlined } from '@ant-design/icons';
// 注册召回组件
interface Values {
  opr: string;
  data: string;
}

interface CollectionCreateFormProps {
  visible: boolean;
  onCreate: (values: Values) => void;
  onCancel: () => void;
}
// 注册召回组件
const RegisterRecall: React.FC<CollectionCreateFormProps> = ({ visible, onCreate, onCancel }) => {
  const [form] = Form.useForm();
  const { TextArea } = Input;
  const text = <span>召回组件名，需要满足正则‘^recall(_([a-z]+))+$</span>;
  return (
    <Modal
      visible={visible}
      title="注册召回组件"
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
      okText="注册"
      cancelText="取消"
    >
      <Form form={form} layout="vertical" name="form_in_modal">
        <Form.Item label="组件名">
          <Row gutter={8}>
            <Col span={22}>
              <Form.Item
                name="component_mark"
                rules={[{ required: true, message: 'Please input the component_mark of collection!' }]}
              >
                <Input style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={2}>
              <Tooltip placement="top" title={text}>
                <QuestionCircleOutlined
                  style={{ color: 'red', fontSize: '18px', marginLeft: '10px', lineHeight: '32px' }}
                />
              </Tooltip>
            </Col>
          </Row>
        </Form.Item>

        <Form.Item rules={[{ required: true, message: '请输入负责人' }]} name="admin" label="负责人">
          <Input />
        </Form.Item>
        <Form.Item name="des" label="描述">
          <TextArea rows={4} />
        </Form.Item>
      </Form>
    </Modal>
  );
};
export default RegisterRecall;
