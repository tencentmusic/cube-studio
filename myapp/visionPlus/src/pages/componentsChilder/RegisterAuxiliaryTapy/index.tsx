import React from 'react';
import { Modal, Form, Input, Select, Tooltip, Row, Col } from 'antd';
import { QuestionCircleOutlined } from '@ant-design/icons';
const { TextArea } = Input;
const { Option } = Select;
interface Values {
  data: {
    component: string;
    component_type: string;
  };
  opr: string;
}

interface CollectionCreateFormProps {
  visible: boolean;
  onCreate: (values: Values) => void;
  onCancel: () => void;
}
// 注册辅助组件类型
const RegisterAuxiliaryTapy: React.FC<CollectionCreateFormProps> = ({ visible, onCreate, onCancel }) => {
  const [form] = Form.useForm();
  const text = <span>组件类型，需要满足‘^组件类别(_([a-z]+))+$</span>;
  const component_typeSelect = ['filter', 'middleware', 'sort'];
  function handleChangeTtypeSelect(value: any) {
    console.log(`selected ${value}`);
  }

  return (
    <Modal
      visible={visible}
      title="注册辅助组件类型"
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
        <Form.Item name="component" label="组件">
          <Select style={{ width: 200 }} onChange={handleChangeTtypeSelect}>
            {component_typeSelect.map(OptionItem => {
              return (
                <Option value={OptionItem} key={OptionItem}>
                  {OptionItem}
                </Option>
              );
            })}
          </Select>
        </Form.Item>

        <Form.Item label="组件类型">
          <Row gutter={8}>
            <Col span={22}>
              <Form.Item
                name="component_type"
                rules={[{ required: true, message: 'Please input the opr of collection!' }]}
              >
                <Input style={{ width: '92%' }} />
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

        <Form.Item name="admin" label="负责人" rules={[{ required: true, message: '请输入负责人' }]}>
          <Input />
        </Form.Item>
        <Form.Item name="des" label="描述">
          <TextArea rows={4} />
        </Form.Item>
      </Form>
    </Modal>
  );
};
export default RegisterAuxiliaryTapy;
