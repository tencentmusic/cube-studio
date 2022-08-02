import React, { useState, useEffect } from 'react';
import { Select, Modal, Form, Input } from 'antd';
import { FormInstance } from 'antd/es/form';
interface Values {
  title: string;
  description: string;
  modifier: string;
}

interface CollectionCreateFormProps {
  visibleOne: boolean;
  onCreateOne: (values: Values) => void;
  onCancel: () => void;
  valueListTableOne: any;
  isomerismFeatureInfoValue: any;
  CONTEXTTYPEValue: any;
  feature_typeShow: any;
  handleChangeInput_descValue: any;
}

const modelOne: React.FC<CollectionCreateFormProps> = ({
  CONTEXTTYPEValue,
  isomerismFeatureInfoValue,
  feature_typeShow,
  valueListTableOne,
  visibleOne,
  onCreateOne,
  onCancel,
  handleChangeInput_descValue,
}) => {
  const [form] = Form.useForm();
  const [FeatureInfoValue, setFeatureInfoValue] = useState([]);
  const formRef1 = React.createRef<FormInstance>();

  function handleChangeInput_desc(value: string) {
    const xxx = formRef1.current;

    const FeatureInfoValue_value = isomerismFeatureInfoValue.filter((item: any) => {
      return item.feature_name === value;
    });
    setFeatureInfoValue(FeatureInfoValue_value);
    xxx?.setFieldsValue({ ctype: '' });
    let hashIdVlaue = '';
    isomerismFeatureInfoValue.forEach((item: any) => {
      if (value === item.feature_name) {
        return (hashIdVlaue = item.hashId);
      }
    });

    handleChangeInput_descValue(hashIdVlaue);
  }

  function handleChange(value: any) {
    let handleVlaue = '';
    const xxx = formRef1.current;
    FeatureInfoValue.map((item: any) => {
      if (value === '2') {
        handleVlaue = item.feature_item_slot;

        return handleVlaue;
      } else if (value === '1') {
        handleVlaue = item.feature_user_slot;
        return handleVlaue;
      }
      return handleVlaue;
    });

    xxx?.setFieldsValue({ raw_fe_name: handleVlaue, dtype: feature_typeShow.label });
  }
  return (
    <Modal
      visible={visibleOne}
      title={valueListTableOne.title}
      okText="确认"
      cancelText="取消"
      onCancel={onCancel}
      onOk={() => {
        form
          .validateFields()
          .then(values => {
            form.resetFields();
            onCreateOne(values);
          })
          .catch(info => {
            console.log('Validate Failed:', info);
          });
      }}
    >
      <Form ref={formRef1} form={form} layout="vertical" initialValues={valueListTableOne.record} name="form_in_modal">
        <Form.Item
          name="input_desc"
          label="特征名字"
          rules={[{ required: true, message: 'Please input the apps of collection!' }]}
        >
          <Select
            showSearch
            optionFilterProp="children"
            filterOption={(input, option: any) => option.children.toLowerCase().indexOf(input.toLowerCase()) >= 0}
            style={{ width: '100%' }}
            onChange={handleChangeInput_desc}
          >
            {isomerismFeatureInfoValue.map((item: any, index: any) => {
              return (
                <Select.Option key={index} value={item.feature_name}>
                  {item.feature_name}
                </Select.Option>
              );
            })}
          </Select>
        </Form.Item>

        <Form.Item rules={[{ required: true, message: '请选择上下文类型' }]} name="ctype" label="上下文类型">
          <Select style={{ width: '100px' }} onChange={handleChange}>
            {CONTEXTTYPEValue.map((item: any) => {
              return (
                <Select.Option key={item.value} value={item.value}>
                  {item.label}
                </Select.Option>
              );
            })}
          </Select>
        </Form.Item>

        <Form.Item name="raw_fe_name" label="特征slot">
          <Input disabled />
        </Form.Item>

        <Form.Item name="dtype" label="特征类型">
          <Input disabled />
        </Form.Item>
      </Form>
    </Modal>
  );
};
export default modelOne;
