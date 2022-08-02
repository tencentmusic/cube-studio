import React, { useState, useEffect } from 'react';
import {
    Button,
    Table, Space,
    Modal,
    message,
    Card,
    Tag
} from 'antd';
import {
    LinkOutlined,
} from '@ant-design/icons';
import QueryForm from "../components/QueryForm/index"
import EditForm from "../components/EditForm/index"
import api from "@src/api/admin"
import { formRolue, form } from "./data.js"
import { STATUS, STATUS_COLOR } from "../common"

import './index.css';

export default function nodeAdmin() {
    const [showEdit, setEditStatus] = useState(false)
    const [tableData, setTableData] = useState([])
    const [chartTableData, setChartTableData] = useState([])
    const [viewForm, setViewForm] = useState<any>({})
    const [isadd, setIsadd] = useState(false)
    const [showChart, setShowChart] = useState(false)

    const nodeColumns = [
        {
            title: '#',
            render: (text: any, record: any, index: number) => <a>{index + 1 + ''}</a>,
        },
        {
            title: "类型",
            dataIndex: 'type',
            key: 'type',
        },
        {
            title: '版本',
            dataIndex: 'version',
            key: 'version',
        },
        {
            title: '工厂名',
            dataIndex: 'factory_name',
            key: 'factory_name',
        },
        {
            title: '名称',
            dataIndex: 'name',
            key: '"name',
        },
        {
            title: '状态',
            dataIndex: 'status',
            key: 'status',
        },
        {
            title: '负责人',
            dataIndex: 'admin',
            key: 'admin',
        },
        {
            title: '描述',
            dataIndex: 'des',
            key: 'des',
        },
        {
            title: '更新时间',
            dataIndex: 'updated_time',
            key: 'updated_time',
        },
    ]



    const columns = [
        {
            title: '#',
            render: (text: any, record: any, index: number) => <a>{index + 1 + ''}</a>,
        },
        {
            title: '节点类型',
            fType: "input",
            dataIndex: 'type',
            key: 'type',
        },
        {
            title: '节点工厂名',
            fType: "input",
            dataIndex: 'factory_name',
            key: 'factory_name',
        },
        {
            title: '状态',
            fType: "input",
            dataIndex: 'status',
            key: '"status',
            render: (text: any) => <Tag color={STATUS_COLOR[text] ? STATUS_COLOR[text] : ""}>{ STATUS[text] ? STATUS[text] : ""}</Tag> 
        },
        {
            title: '文档地址',
            fType: "input",
            dataIndex: 'url',
            key: 'url',
            width: 100,
            render: (text: any) => <LinkOutlined onClick={() => window.open(text, "blank")} />
        },
        {
            title: '描述',
            fType: "textArea",
            dataIndex: 'des',
            key: 'des',
        },
        {
            title: '负责人',
            fType: "input",
            dataIndex: 'admin',
            key: 'admin',
        },
        {
            title: '更新时间',
            dataIndex: 'updated_time',
            key: 'updated_time',
        },
        {
            title: '操作',
            key: 'action',
            render: (text: any, record: any) => (
                <Space size="middle">
                    <a key="view" onClick={() => {

                        setIsadd(false)
                        setViewForm({ ...record })
                        setEditStatus(true)
                    }}>查看</a>
                    {/* <a key="edit" onClick={() => {
                        console.log(record)
                        setIsadd(true)
                        setViewForm({ ...record })
                        setEditStatus(true)

                    }}>编辑</a>
                    <Popconfirm title="是否删除?" okText="确认" cancelText="取消" onConfirm={() => {
                        deleteNode(record)
                    }}>
                        <a key="del">删除</a>
                    </Popconfirm>
                    <a key="node" onClick={() => { relationNode(record) }}> 历史版本 </a> */}
                </Space>
            ),
        },
    ];

    const viewFormRolue = columns.filter((item: any) => item.fType).map((item: any) => {
        return {
            type: item.fType,
            name: item.dataIndex,
            label: item.title
        }
    })




    const queryList = (params: any) => {
        const { admin, factory_name, status, time, type } = params
        const data = {
            factory_name: factory_name || "",
            status: status || 0, // 负责人
            admin: admin || "",
            type: type || "",
            min_updated_time: time && time[0] ? new Date(time[0]).getTime() : 0,
            max_updated_time: time && time[1] ? new Date(time[1]).getTime() : 0,
        }
        api.query_node_factorys({ opr: "query_node_factorys", data }).then(res => {
            if (res.status === 0) {
                setTableData(res.data.results)
                message.success(res.message);
            } else {
                message.error(res.message);
            }
        })
    }

    const submitForm = (obj: any) => {
        const data: any = {
            ...obj
        }
        if (viewForm.version) {
            data.version = viewForm.version + 1
            api.mod_scene({ opr: "mod_scene", data }).then(res => {
                if (res.status === 0) {
                    queryList({})
                    setEditStatus(false)
                    message.success(res.message);
                } else {
                    message.error(res.message);
                }
            })
        } else {
            api.add_scene({ opr: "add_scene", data }).then(res => {
                if (res.status === 0) {
                    queryList({})
                    setEditStatus(false)
                    message.success(res.message);
                } else {
                    message.error(res.message);
                }
            })
        }
    }


    useEffect(() => {
        queryList({})
    }, [])

    const resFied = () => {
        setEditStatus(false)
        queryList({})
    }


    return (
        <div style={{ padding: "30px 40px" }}>
            <div>
                <Card title="查询条件">
                    <QueryForm formRolue={formRolue} form={form} queryList={queryList}></QueryForm>
                </Card>
                <Card title="查询结果" style={{ marginTop: "20px" }}>
                    <Table key="scene_name" columns={columns} dataSource={tableData} />
                </Card>
            </div>
            <Modal width={900} title={isadd ? isadd && viewForm.version ? "编辑" : "新增" : "查看"} visible={showEdit} onCancel={() => { setEditStatus(false) }} footer={[
                <Button key="back" onClick={() => {
                    setEditStatus(false)
                }}>关闭</Button>
            ]}>
                <EditForm formRolue={viewFormRolue} form={viewForm} isadd={isadd} showEdit={showEdit} submitForm={submitForm} resFied={resFied} />
            </Modal>

            <Modal width={"60%"} title="关联图" visible={showChart} onCancel={() => { setShowChart(false) }} footer={[
                <Button key="back" onClick={() => {
                    setShowChart(false)
                }}>关闭</Button>
            ]}>
                <Table key="struct_name" columns={nodeColumns} dataSource={chartTableData} />
            </Modal>

        </div>
    );
}
