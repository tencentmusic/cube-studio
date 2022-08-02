import React, { useState, useEffect, useRef } from 'react';
import { Form, Input, Button, Checkbox, Select, DatePicker, Radio } from 'antd';
import ReactJson from 'react-json-view';
import api from "@src/api/admin"
const { TextArea } = Input;




export default function EditForm(props: any) {
    const [form] = Form.useForm();
    const [jsonStr, setJsonStr] = useState<any>({})


    const formRef = useRef(null)
    // useEffect(()=>{
    //     console.log("触发了", (formRef.current as any).setFieldsValue(props.form))
    // }, [props.nodeId, props.form.factory_name])



    useEffect(() => {
        form.resetFields();
        form.setFieldsValue(props.form);
        if (props.form.map_config) {
            setJsonStr(JSON.parse(JSON.stringify(props.form.map_config) ))
        }
        if(props.saveFrom && props.saveFrom.graph_name) {
            form.setFieldsValue(props.saveFrom);
        }
    }, [])

    const onFinishFailed = () => {
        console.log("失败")
    }
    const onFinish = (value: any) => {
        props.submitForm(value, props.opName)
    }
    const getEdit = (e: any) => {
        console.log(e)
        return true
    }
    const getAdd = (e: any) => {
        console.log(e)
        return true
    }
    return (
        <div>
            <Form
                name="basic"
                labelCol={{ span: 8 }}
                style={{ width: "80%" }}
                wrapperCol={{ span: 16 }}
                initialValues={{ remember: true }}
                onFinish={onFinish}
                onFinishFailed={onFinishFailed}
                autoComplete="off"
                ref={formRef}
                form={form}
            >
                {
                    props.formRolue.map((item: any) => {
                        switch (item.type) {
                            case "input":
                                return (<Form.Item
                                    key={item.name}
                                    label={item.label}
                                    name={item.name}
                                    initialValue={props.form[item.name]}
                                    rules={[{ required: true, message: '请输入' + item.label }]}
                                >
                                    <Input disabled={props.disableds || props.isadd === false ? true : false} />
                                </Form.Item>)
                                break;
                            case "select":
                                return (
                                    <Form.Item
                                        key={item.name}
                                        label={item.label}
                                        name={item.name}
                                        rules={[{ required: true, message: '请选择' + item.label }]}
                                    >
                                        <Select disabled={props.disableds || props.isadd === false ? true : false}>
                                            {
                                                item.options ? item.options.map((select: any) => {
                                                    return <Select.Option key={select.value} value={select.value}>{select.label}</Select.Option>
                                                }) : null
                                            }
                                        </Select>
                                    </Form.Item>
                                )
                                break;
                            case "picker":
                                return (
                                    <Form.Item
                                        key={item.name}
                                        label={item.label}
                                        name={item.name}
                                        rules={[{ required: true, message: '请选择' + item.label }]}
                                    >
                                        <DatePicker style={{ width: '100%' }} />
                                    </Form.Item>
                                )
                                break;
                            case "radio":
                                return (
                                    <Form.Item
                                        key={item.name}
                                        label={item.label}
                                        name={item.name}
                                        rules={[{ required: true, message: '请选择' + item.label }]}
                                    >
                                        <Radio.Group disabled={props.disableds || props.isadd === false ? true : false}>
                                            {/* <Radio value={1}>A</Radio> */}
                                            {
                                                item.options ? item.options.map((radio: any) => {
                                                    return <Radio key={radio.value} value={radio.value}>{radio.label}</Radio>
                                                }) : null
                                            }
                                        </Radio.Group>
                                    </Form.Item>
                                )
                                break;
                            case "json":
                                return (
                                    <Form.Item
                                        key={item.name}
                                        label={item.label}
                                        name={item.name}
                                    // rules={[{ required: true, message: '请输入' + item.label }]}
                                    >
                                        <ReactJson
                                           // eslint-disable-next-line
                                            src={ (jsonStr)  }
                                            displayDataTypes={false}
                                            theme="railscasts"
                                            // name={item.name}
                                            onEdit={e => getEdit(e)}
                                            onAdd={e => getAdd(e)}
                                        />
                                    </Form.Item>
                                )
                                break;
                            case "textArea":
                                return (
                                    <Form.Item
                                        key={item.name}
                                        label={item.label}
                                        name={item.name}
                                        rules={[{ required: true, message: '请输入' + item.label }]}
                                    >
                                        <TextArea rows={4} disabled={props.disableds || props.isadd === false ? true : false} />
                                    </Form.Item>
                                )
                                break;
                            default:
                                null
                        }
                    })
                }
                <Form.Item wrapperCol={{ offset: 8, span: 16 }}>
                    <div style={{ display: "flex", justifyContent: 'center' }}>
                        {
                            props.stepCurrent === 0 ? <Button onClick={() => {
                                props.setEditStatus(false)
                                props.setBaseForm ? props.setBaseForm({}) : ""
                            }}>
                                返回
                    </Button> : null
                        }

                        {
                            props.stepCurrent === 0 ? <Button htmlType="submit" style={{ marginLeft: "20px" }} >
                                下一步
                    </Button> : null
                        }
                    </div>
                    {
                        props.stepCurrent && props.stepCurrent !== 0 ? <Button type="primary">
                            提交
                     </Button> : null
                    }
                    {
                        props.isadd &&  props.stepCurrent === undefined ? <Button htmlType="submit" type="primary" disabled={props.disableds || props.isadd === false ? true : false}>
                            提交
                     </Button> : null
                    }

                </Form.Item>
            </Form>
        </div>
    )
}