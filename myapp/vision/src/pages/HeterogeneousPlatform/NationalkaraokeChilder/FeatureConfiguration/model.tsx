import React, { useState, useEffect } from 'react';
import { Select, Modal, Form, Input } from 'antd';
import { FormInstance } from 'antd/es/form';
interface Values {
  title: string;
  description: string;
  modifier: string;
}

interface CollectionCreateFormProps {
  visible: boolean;
  onCreate: (values: Values) => void;
  onCancel: () => void;
  setshowregister: (value: any) => void;
  valueListTable: any;
  featureDisplayOpValue: any;
  statusOPTIONS: any;
  FEATURETYPE: any;
  Datalist: any;
}

const model: React.FC<CollectionCreateFormProps> = ({
  featureDisplayOpValue,
  statusOPTIONS,
  FEATURETYPE,
  valueListTable,
  visible,
  onCreate,
  onCancel,
  setshowregister,
  Datalist,
}) => {
  const [form] = Form.useForm();
  const [param_count, setParam_count] = useState([]);
  const formRef1 = React.createRef<FormInstance>();

  function handleChange(value: any) {
    featureDisplayOpValue.map((item: any) => {
      if (item.op_name === value) {
        return setshowregister(item.op_id);
      }
    });
    setParam_count(value);
  }

  const FormItem = featureDisplayOpValue.map((item: any) => {
    if (item.op_name === param_count) {
      const param_typeArray = item.param_type.split(',');

      return param_typeArray.map((item: any, index: number) => {
        if (item === '1') {
          return (
            <Form.Item label={`参数${index + 1}`} name={`parameter${index + 1}`}>
              <Select style={{ width: '100%' }}>
                {Datalist.map((item: any) => {
                  return (
                    <Select.Option key={item.input_item_id} value={item.raw_fe_name}>
                      {'_' + item.raw_fe_name}
                    </Select.Option>
                  );
                })}
              </Select>
            </Form.Item>
          );
        } else {
          return (
            <Form.Item key={index} label={`参数${index + 1}`} name={`parameter${index + 1}`}>
              <Input />
            </Form.Item>
          );
        }
      });
    }
  });
  useEffect(() => {
    if (valueListTable.record.params) {
      const param_typeArray = valueListTable.record.params.split(',');

      for (let i = 0; i < param_typeArray.length; i++) {
        return formRef1.current?.setFieldsValue({
          [`parameter${i}`]: param_typeArray[i],
        });
      }
    }
  });
  useEffect(() => {
    console.log(valueListTable.record, '7777');

    formRef1.current?.setFieldsValue({
      ...valueListTable.record,
      parameter1: valueListTable.record.params ? valueListTable.record.params.split(',')[0] : '',
      parameter2: valueListTable.record.params ? valueListTable.record.params.split(',')[1] : '',
      parameter3: valueListTable.record.params ? valueListTable.record.params.split(',')[2] : '',
      parameter4: valueListTable.record.params ? valueListTable.record.params.split(',')[3] : '',
      parameter5: valueListTable.record.params ? valueListTable.record.params.split(',')[4] : '',
      parameter6: valueListTable.record.params ? valueListTable.record.params.split(',')[5] : '',
    });
    if (valueListTable.record.op_item_name) {
      setParam_count(valueListTable.record.op_item_name);
    }
  }, [valueListTable.record]);

  return (
    <Modal
      visible={visible}
      title={valueListTable.title}
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
      <Form
        ref={formRef1}
        form={form}
        layout="vertical"
        initialValues={{
          ...valueListTable.record,

          parameter1: valueListTable.record.params ? valueListTable.record.params.split(',')[0] : '',
          parameter2: valueListTable.record.params ? valueListTable.record.params.split(',')[1] : '',
          parameter3: valueListTable.record.params ? valueListTable.record.params.split(',')[2] : '',
          parameter4: valueListTable.record.params ? valueListTable.record.params.split(',')[3] : '',
          parameter5: valueListTable.record.params ? valueListTable.record.params.split(',')[4] : '',
          parameter6: valueListTable.record.params ? valueListTable.record.params.split(',')[5] : '',
        }}
        name="form_in_modal"
      >
        <Form.Item
          name="op_item_name"
          label="算子名字"
          rules={[{ required: true, message: 'Please input the apps of collection!' }]}
        >
          <Select style={{ width: '100%' }} onChange={handleChange}>
            {featureDisplayOpValue.map((item: any) => {
              return (
                <Select.Option key={item.op_id} value={item.op_name}>
                  {item.op_name}
                </Select.Option>
              );
            })}
          </Select>
        </Form.Item>

        <Form.Item rules={[{ required: true, message: '请输入特征名字' }]} name="rst_fe_name" label="特征名字">
          <Input />
        </Form.Item>

        {FormItem}
        {/* <Form.Item label="参数1" name="parameterOne">
          <Input />
        </Form.Item>
        <Form.Item name="parameterTwo" label="参数2">
          <Input />
        </Form.Item>
        <Form.Item name="parameterThree" label="参数3">
          <Input />
        </Form.Item> */}

        <Form.Item rules={[{ required: true, message: '请输入输出特征slot' }]} name="out_index" label="输出特征slot">
          <Input style={{ width: '100%' }} />
        </Form.Item>
        <Form.Item
          name="out_fe_type"
          label="输出类型"
          rules={[{ required: true, message: 'Please input the apps of collection!' }]}
        >
          <Select style={{ width: '100%' }}>
            {FEATURETYPE.map((item: any) => {
              return (
                <Select.Option key={item.value} value={item.value}>
                  {item.label}
                </Select.Option>
              );
            })}
          </Select>
        </Form.Item>
      </Form>
    </Modal>
  );
};
export default model;
