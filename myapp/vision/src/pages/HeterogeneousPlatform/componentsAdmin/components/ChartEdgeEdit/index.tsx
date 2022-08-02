import React, { useState, useEffect } from 'react';
import { Form, Input, Button, Select, Radio, Modal } from 'antd';
import {
    CopyOutlined,
    QuestionCircleOutlined,
    EditOutlined
} from '@ant-design/icons';
import { copyStr } from "../ChartNodeEdit/fun"
import "./index.css"
import api from "@src/api/admin"
const { TextArea } = Input;




export default function chartEdit(props: any) {
    const [form] = Form.useForm();
    const [formModal] = Form.useForm();
    const [isModalVisible, setIsModalVisible] = useState(false)
    const [jsonStr, setJsonStr] = useState<any>({})
    const [isDes, setIsDes] = useState(false)
    const [isNew, setIsNew] = useState(false)


    useEffect(() => {
        console.log("edgeName", props.edgeName)
        form.resetFields();
        if (props.edgeName.edgeName) {

            if (props.edgeName.factory_name) {
                form.setFieldsValue(props.edgeName)
            } else {
                const item1 = props.nodeInfoArr.
                    filter(((item: any) => item.type === "edge")).
                    find((item: any) => props.edgeName.edgeName === item.name)
                if (item1) {
                    setIsNew(false)
                    form.setFieldsValue(item1)
                } else {
                    form.setFieldsValue({
                        ...props.edgeName,
                        name: props.edgeName.edgeName
                    })
                    setIsNew(true)
                }
            }

        }
    }, [props.edgeName.edgeName])

    const onFinishFailed = () => {
        console.log("失败")
    }
    const onFinish = (value: any) => {
        props.edgeSubmit && props.edgeSubmit(value, props.edgeName)
    }
    const copyText = (name: string) => {
        const values = form.getFieldsValue(true)
        copyStr(values[name] !== undefined ? values[name] : "")
    }

    const okData = () => {
        if (isDes) {
            form.setFieldsValue({
                des: formModal.getFieldsValue(['des'])["des"] ? formModal.getFieldsValue(['des'])["des"] : ""
            })
        } else {
            form.setFieldsValue({
                config: formModal.getFieldsValue(['config'])["config"] ? formModal.getFieldsValue(['config'])["config"] : ""
            })
        }
        setIsModalVisible(false)
    }
    const showModel = (name: string) => {
        console.log(123123, name)
        if(name === "config") {
            const data = {
                opr: "get_config_prototype", 
                data: {
                    type:"edge",
                    factory_name:form.getFieldsValue(['factory_name'])["factory_name"] ? form.getFieldsValue(['factory_name'])["factory_name"]: "",
                    template_name: props.baseForm.template_name
                }
            }
            api.get_config_prototype(data).then(res => {
                 if(res.status === 0 ) {
                     formModal.setFieldsValue({config: res.data.config})
                 }
            })
        }
        setIsModalVisible(true);
        console.log(form.getFieldsValue([name]))
        formModal.setFieldsValue({
            factory_name: form.getFieldsValue(['factory_name'])["factory_name"] ? form.getFieldsValue(['factory_name'])["factory_name"] : "",
            des: form.getFieldsValue(['des'])["des"] ? form.getFieldsValue(['des'])["des"] : "",
            config: form.getFieldsValue(['config'])["config"] ? form.getFieldsValue(['config'])["config"] : ""
        })
        setIsDes(name === "des" ? true : false)
        
    }

    return (
        <div style={{ position: "relative" }} >
            <Form
                name="basic"
                labelCol={{ span: 8 }}
                style={{ width: "80%" }}
                wrapperCol={{ span: 16 }}
                initialValues={{ remember: true }}
                onFinish={onFinish}
                onFinishFailed={onFinishFailed}
                autoComplete="off"
                form={form}
                className="abtset-form "
            >

                <Form.Item
                    label="配置类型"
                    name="a"
                >
                    <Radio.Group defaultValue={2} disabled={true}>
                        <Radio value={1}>共有</Radio>
                        <Radio value={2}>私有</Radio>
                    </Radio.Group>
                </Form.Item>
                <Form.Item
                    label="边工厂"
                    name="factory_name"
                    rules={[{ required: true, message: '请选择边工厂' }]}
                >
                    {/* <Input></Input>  */}
                    <Select disabled={props.operation === "view"}>

                        {
                            props.edgeInfoArr.map((item: any) => (
                                <Select.Option key={item} value={item}>{item}</Select.Option>
                            ))
                        }
                    </Select>
                    {/* <CopyOutlined className="input-icon-copy"/> */}
                </Form.Item>
                <Form.Item
                    label="边名称"
                    name="name"
                    rules={[{ required: true, message: '请选择边名称' }]}
                >
                    <Input disabled={props.operation === "view"}></Input>
                </Form.Item>
                <Form.Item
                    label="负责人"
                    name="admin"
                    rules={[{ required: true, message: '请输入负责人' }]}
                >
                    <Input disabled={props.operation === "view"}></Input>
                </Form.Item>
                <Form.Item
                    label="配置"
                    name="config"
                    rules={[{ required: true, message: '请选择配置' }]}
                >
                    <TextArea rows={4} disabled={true} />
                    {/* <CopyOutlined className="input-icon-copy"/>  */}
                    {/* <EditOutlined className="input-icon-edit"/> */}
                </Form.Item>
                <Form.Item
                    label="边说明"
                    name="des"
                    rules={[{ required: true, message: '请选择边说明' }]}
                >
                    <TextArea rows={4} disabled={true} />
                    {/* <CopyOutlined className="input-icon-copy"/>  */}
                    {/* <EditOutlined className="input-icon-edit"/> */}
                </Form.Item>

                <Form.Item wrapperCol={{ offset: 8, span: 16 }}>
                    <Button htmlType="submit" disabled={props.operation === "view"} type="primary">保存</Button>
                </Form.Item>
            </Form>
            <div className="sc-box1">

                <CopyOutlined className="input-icon-copy1" onClick={() => copyText("factory_name")} style={{ top: "64px" }} />

                <QuestionCircleOutlined className="input-icon-copy1" style={{ top: "110px" }} />
                <CopyOutlined className="input-icon-copy1" onClick={() => copyText("name")} style={{ top: "130px" }} />


                <CopyOutlined className="input-icon-copy1" onClick={() => copyText("config")} style={{ top: "248px" }} />
                <EditOutlined className="input-icon-edit1" onClick={() => { showModel('config') }} style={{ top: "278px" }} />
                <CopyOutlined className="input-icon-copy1" onClick={() => copyText("des")} style={{ top: "380px" }} />
                <EditOutlined className="input-icon-edit1" onClick={() => { showModel('des') }} style={{ top: "410px" }} />
            </div>


            <Modal title="编辑" width={"60%"} visible={isModalVisible} onOk={() => okData()} onCancel={() => { setIsModalVisible(false); setJsonStr({}) }} okText="确认" cancelText="取消">
                <Form
                    name="basic"
                    labelCol={{ span: 8 }}
                    style={{ width: "80%" }}
                    wrapperCol={{ span: 16 }}
                    // initialValues={{ remember: true }}
                    onFinish={onFinish}
                    onFinishFailed={onFinishFailed}
                    autoComplete="off"
                    form={formModal}
                >

                    <Form.Item
                        label="名称"
                        name="factory_name"
                    >
                        <Input></Input>
                    </Form.Item>
                    {
                        isDes ?
                            <Form.Item
                                label="描述信息"
                                name="des"
                            >
                                <TextArea rows={4} />
                            </Form.Item> :
                            <Form.Item
                                label="配置信息"
                                name="config"
                            >
                                <TextArea rows={4} />
                                {/* <ReactJson
                                    src={jsonStr}
                                    displayDataTypes={false}
                                    theme="railscasts"
                                    // name='config'
                                    onEdit={e => getEdit(e)}
                                    onAdd={e => getAdd(e)}
                                /> */}
                            </Form.Item>
                    }
                </Form>
            </Modal>
        </div>
    )
}