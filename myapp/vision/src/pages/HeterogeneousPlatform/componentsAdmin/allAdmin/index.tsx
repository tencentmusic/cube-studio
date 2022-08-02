import React, { useState, useEffect } from 'react';
import {
    Button,
    Table, Space, Popconfirm,
    Modal,
    message,
    Card,
    Tag
} from 'antd';
import QueryForm from "../components/QueryForm/index"
import EditForm from "../components/EditForm/index"
import api from "@src/api/admin"
import { formRolue, form } from "./data.js"
import { STATUS, STATUS_COLOR } from "../common"
import './index.css';

export default function allAdmin() {
    const [showEdit, setEditStatus] = useState(false)
    const [tableData, setTableData] = useState([])
    const [chartTableData, setChartTableData] = useState([])
    const [viewForm, setViewForm] = useState<any>({})
    const [isadd, setIsadd] = useState(false)
    const [showChart, setShowChart] = useState(false)
    const [isChart, setIsChart] = useState(false)

    const deleteNode = (record: any) => {
        const data = {
            opr: "delete_component",
            data: {
                type: record.type,
                factory_name: record.factory_name,
                name: record.name,
            }
        }
        api.delete_component(data).then(res => {
            console.log(res)
            if (res.status === 0) {
                message.success(res.message);
            } else {
                message.error(res.message);
            }
        })
    }
    const relationNode = (record: any) => {
        setIsChart(false)
        const data = {
            opr: "query_associated_graphs",
            data: {
                associated_graphs: record.associated_graphs,
            }
        }
        api.query_associated_graphs(data).then(res => {
            console.log(res)
            if (res.status === 0) {
                setChartTableData(res.data.results)
                setShowChart(true)
                message.success(res.message);
            } else {
                message.error(res.message);
            }
        })
    }
    const historyNode = (record: any) => {
        setIsChart(true)
        const data = {
            opr: "query_component_historys",
            data: {
                name: record.name,
                is_public: record.status === 4 ? 1 : 0,
                graph_name: record.graph_name
            }
        }
        api.query_component_historys(data).then(res => {
            console.log(res)
            if (res.status === 0) {
                setChartTableData(res.data.results)
                setShowChart(true)
                message.success(res.message);
            } else {
                message.error(res.message);
            }
        })
    }


    const nodeColumns = [
        {
            title: '#',
            render: (text: any, record: any, index: number) => <a>{index + 1 + ''}</a>,
        },
        {
            title: "图名称",
            dataIndex: 'graph_name',
            key: 'graph_name',
        },
        {
            title: '版本',
            dataIndex: 'version',
            key: 'version',
        },
        {
            title: '图模板',
            dataIndex: 'template_name',
            key: 'template_name',
        },
        {
            title: '负责人',
            dataIndex: 'admin',
            key: 'admin',
        },
        // {
        //     title: '描述',
        //     dataIndex: 'des',
        //     key: 'des',
        // },
        {
            title: '更新时间',
            dataIndex: 'updated_time',
            key: 'updated_time',
        },
        {
            title: '使用场景',
            dataIndex: 'scene_name',
            key: '"scene_name',
        },
        {
            title: '操作',
            key: 'action',
            render: (text: any, record: any) => (
                <Space size="middle">
                    {/* <a key="view" onClick={() => {
                    }}>查看</a> */}
                    <span style={{color: "#999"}}> 查看 </span>
                </Space>
            ),
        },
    ]







    const nodeColumns1 = [
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
            render: (text: any) => <Tag color={STATUS_COLOR[text] ? STATUS_COLOR[text] : ""}>{ STATUS[text] ? STATUS[text] : ""}</Tag> 
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
            title: '类型',
            fType: "select",
            foptions: [
                {
                    label: "recall",
                    value: "recall"
                },
                {
                    label: "sort",
                    value: "sort"
                },
                {
                    label: "filter",
                    value: "filter"
                },
            ],
            dataIndex: 'type',
            key: 'type',
        },
        {
            title: '工厂名',
            fType: "input",
            dataIndex: 'factory_name',
            key: 'factory_name',
        },
        {
            title: '名称',
            fType: "input",
            dataIndex: 'name',
            key: '"name',
        },
        {
            title: '输出结构名',
            fType: "input",
            dataIndex: 'output_struct',
            key: 'output_struct',
        },
        {
            title: '状态',
            fType: "select",
            foptions: [
                {
                    value: -1,
                    label: '失效'
                },
                {
                    value: 0,
                    label: "有效"
                },
                {
                    value: 1,
                    label: "创建成功"
                },
                {
                    value: 2,
                    label: '测试发布'
                },
                {
                    value: 3,
                    label: '正式发布'
                },
                {
                    value: 4,
                    label: '公有'
                },
                {
                    value: 5,
                    label: '私有'
                },
            ],
            dataIndex: 'status',
            key: 'status',
            render: (text: any) => <Tag color={STATUS_COLOR[text] ? STATUS_COLOR[text] : ""}>{ STATUS[text] ? STATUS[text] : ""}</Tag> 
        },
        // {
        //     title: '描述',
        //     fType: "textArea",
        //     dataIndex: 'des',
        //     key: 'des',
        // },
        {
            title: '负责人',
            fType: "input",
            dataIndex: 'admin',
            key: 'admin',
        },
        {
            title: '更新时间',
            fType: "input",
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
                    <span style={{ color: "#999", marginTop: "-1px" }}> | </span>
                    <Popconfirm title="是否删除?" okText="确认" cancelText="取消" onConfirm={() => {
                        deleteNode(record)
                    }}>
                        <a key="del">删除</a>
                    </Popconfirm>
                    <span style={{ color: "#999", marginTop: "-1px" }}> | </span>
                    <a key="node" onClick={() => { historyNode(record) }}> 历史版本 </a>
                    <span style={{ color: "#999", marginTop: "-1px" }}> | </span>
                    <a key="node" onClick={() => { relationNode(record) }}> 关联图 ({ record.associated_graphs && record.associated_graphs.length ? record.associated_graphs.length : 0 })</a>
                </Space>
            ),
        },
    ];

    const viewFormRolue = columns.filter((item: any) => item.fType).map((item: any) => {
        return {
            type: item.fType,
            name: item.dataIndex,
            label: item.title,
            options: item.foptions ? item.foptions : []
        }
    })




    const queryList = (params: any) => {
        const { admin, factory_name, type, name, time, status } = params
        const data = {
            type: type || "",
            factory_name: factory_name || "",
            name: name || "",
            status: status || 0,
            admin: admin || "",
            min_updated_time: time && time[0] ? new Date(time[0]).getTime() : 0,
            max_updated_time: time && time[1] ? new Date(time[1]).getTime() : 0,
        }
        api.query_components({ opr: "query_components", data }).then(res => {
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

            <Modal width={"80%"} title="关联图" visible={showChart} onCancel={() => { setShowChart(false) }} footer={[
                <Button key="back" onClick={() => {
                    setShowChart(false)
                }}>关闭</Button>
            ]}>
                <Table key="struct_name" columns={isChart ? nodeColumns1 : nodeColumns} dataSource={chartTableData} />
            </Modal>

        </div>
    );
}
