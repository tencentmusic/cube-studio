import React, { useState, useEffect, useRef } from 'react';
import { Form, Input, Button, Checkbox, Select, DatePicker, Radio, Modal } from 'antd';
import ReactJson from 'react-json-view';
import {
    CopyOutlined,
    QuestionCircleOutlined,
    EditOutlined
} from '@ant-design/icons';
import api from "@src/api/admin"
import "./index.css"
import { copyStr } from "./fun"
const { TextArea } = Input;



export default function chartEdit(props: any) {
    const [form] = Form.useForm();
    const [formModal] = Form.useForm();
    const [node_factory_names, setNode_factory_names] = useState<any>([])
    const [isModalVisible, setIsModalVisible] = useState(false)
    const [jsonStr, setJsonStr] = useState<any>({})
    const [isDes, setIsDes] = useState(false)
    const [isNew, setIsNew] = useState(false)

    useEffect(() => {
        form.resetFields();
        if (props.form.name) {
            if (props.form.des) {
                form.setFieldsValue(props.form)
            } else {
                const item1 = props.nodeInfoArr.
                    filter(((item: any) => item.type !== "edge")).
                    find((item: any) => props.form.name === item.name)
                if (item1) {
                    setIsNew(false)
                    form.setFieldsValue(item1)
                } else {
                    form.setFieldsValue({
                        factory_name: props.form.name,
                        type: props.form.typeName ? props.form.typeName : ""
                    })
                    setIsNew(true)
                }
            }

        }
    }, [props.nodeId])
    
     
    const onFinishFailed = () => {
        console.log("失败")
    }
    const onFinish = (value: any) => {
        props.nodeSubmit && props.nodeSubmit({ form: props.form, value: value })
    }

    const typeChange = (e: any) => {
        form.setFieldsValue({ factory_name: "" })
        api.get_node_factory_names({ opr: "get_node_factory_names", data: { type: e } }).then(res => {
            if (res.status === 0 && res.data.node_factory_names) {
                setNode_factory_names(res.data.node_factory_names)
            }
        })
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

    const getEdit = (e: any) => {
        return true
    }

    const getAdd = (e: any) => {
        return true
    }

    const showModel = (name: string) => {
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
        <div>
            <Form
                name="basic"
                labelCol={{ span: 8 }}
                style={{ width: "80%" }}
                wrapperCol={{ span: 16 }}
                // initialValues={{ remember: true }}
                onFinish={onFinish}
                onFinishFailed={onFinishFailed}
                autoComplete="off"
                form={form}
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
                    label="执行等级"
                    name="level"
                    rules={[{ required: true, message: '请选择执行等级' }]}
                >
                    <Radio.Group   disabled={props.operation === "view"} >
                        <Radio value={1}>一般</Radio>
                        <Radio value={0}>重要</Radio>
                    </Radio.Group>
                </Form.Item>
                <Form.Item
                    label="节点类型"
                    name="type"
                    rules={[{ required: true, message: '请选择节点类型' }]}
                >
                    <Select disabled={props.operation === "view"} onChange={(e) => {
                        typeChange(e)
                    }}>
                        <Select.Option value="recall">recall</Select.Option>
                        <Select.Option value="sort">sort</Select.Option>
                        <Select.Option value="filter">filter</Select.Option>
                    </Select>
                </Form.Item>
                <Form.Item
                    label="节点工厂"
                    name="factory_name"
                    rules={[{ required: true, message: '请选择节点工厂' }]}
                >
                    <Select disabled={props.operation === "view"}>
                        {node_factory_names.map((item: any) => (
                            <Select.Option key={item} value={item}>{item}</Select.Option>
                        ))}
                    </Select>
                </Form.Item>
                <Form.Item
                    label="节点名称"
                    name="name"
                    rules={[{ required: true, message: '请选择节点名称' }]}
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
                    label="节点配置"
                    name="config"
                    rules={[{ required: true, message: '请选择节点配置' }]}
                >
                    <TextArea rows={4} disabled={true} />
                </Form.Item>
                <Form.Item
                    label="节点描述"
                    name="des"
                    rules={[{ required: true, message: '请选择节点描述' }]}
                >
                    <TextArea rows={4} disabled={true} />
                </Form.Item>

                <Form.Item wrapperCol={{ offset: 8, span: 16 }}>
                    <Button disabled={props.operation === "view"} htmlType="submit" type="primary">保存</Button>
                </Form.Item>
            </Form>
            <div className="sc-box">
                <CopyOutlined className="input-icon-copy" onClick={() => copyText("type")} style={{ top: "188px" }} />
                <CopyOutlined className="input-icon-copy" onClick={() => copyText("factory_name")} style={{ top: "245px" }} />
                <CopyOutlined className="input-icon-copy" onClick={() => copyText("name")} style={{ top: "308px" }} />
                <QuestionCircleOutlined className="input-icon-copy" style={{ top: "290px" }} />
                <CopyOutlined className="input-icon-copy" onClick={() => copyText("config")} style={{ top: "425px" }} />
                <EditOutlined className="input-icon-edit" onClick={() => { showModel('config') }} style={{ top: "460px" }} />
                <CopyOutlined className="input-icon-copy" onClick={() => copyText("des")} style={{ top: "552px" }} />
                <EditOutlined className="input-icon-edit" onClick={() => { showModel('des') }} style={{ top: "578px" }} />
            </div>


            <Modal title="Basic Modal" width={"60%"} visible={isModalVisible} onOk={() => okData()} onCancel={() => { setIsModalVisible(false); setJsonStr({}) }} okText="确认" cancelText="取消">
                <Form
                    name="basic"
                    labelCol={{ span: 8 }}
                    style={{ width: "80%" }}
                    wrapperCol={{ span: 16 }}
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