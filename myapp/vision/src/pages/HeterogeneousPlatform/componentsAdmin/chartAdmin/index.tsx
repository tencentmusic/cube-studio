import React, { useState, useEffect } from 'react';
import {
    Button,
    Table, Space, Modal, message, Card,
    Tag
} from 'antd';
import StepsForm from "../components/StepsForm/index"
import QueryForm from "../components/QueryForm/index"
import api from "@src/api/admin"
import { formRolue } from "./data.js"
import { STATUS, STATUS_COLOR } from "../common"
import {useHistory} from "react-router-dom"
import './index.css';

export default function chartAdmin() {
    const [showEdit, setEditStatus] = useState(false)
    const [tableData, setTableData] = useState([])
    const [record, setRecord] = useState({})
    const [template_names, setTemplate_names] = useState([])
    const [isadd, setIsadd] = useState(false)
    const [scene_names, setScene_names] = useState([])
    const [showChart, setShowChart] = useState(false)
    const [chartTableData, setChartTableData] = useState([])
    const [operation, setOperation] = useState('')

    const history = useHistory()

    const itemView = (record: any) => {
        setRecord(record)
        setEditStatus(true)
    }
    const nodeColumns1 = [
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
                    <a key='view' onClick={()=>{
                         setIsadd(false)
                         itemView(record)
                         setOperation("view")
                         setShowChart(false)
                    }}> 查看 </a>
                </Space>
            ),
        },
    ]



    const historyNode = (record: any) => {
        const data = {
            opr: "query_graph_historys",
            data: {
                graph_name: record.graph_name,
            }
        }
        api.query_graph_historys(data).then(res => {
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

    const rollback_graph = (record: any) => {
        const data = {
            opr: "rollback_graph",
            data: {
                graph_name: record.graph_name,
                version: record.version,
            }
        }
        api.rollback_graph(data).then(res => {
            console.log(res)
            if (res.status === 0) {
                message.success(res.message);
            } else {
                message.error(res.message);
            }
        })
    }

    const columns = [
        {
            title: '#',
            render: (text: any, record: any, index: number) => <a>{index + 1 + ''}</a>,
        },
        {
            title: '图名称',
            dataIndex: 'graph_name',
            key: 'graph_name',
        },
        {
            title: '版本',
            dataIndex: 'version',
            key: 'version',
        },
        {
            title: '状态',
            dataIndex: 'status',
            key: 'status',
            render: (text: any) => <Tag color={STATUS_COLOR[text] ? STATUS_COLOR[text] : ""}>{ STATUS[text] ? STATUS[text] : ""}</Tag> 
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
        {
            title: '描述',
            dataIndex: 'des',
            key: 'des',
        },
        {
            title: '使用场景',
            dataIndex: 'scene_name',
            key: 'scene_name',
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
                        itemView(record)
                        setOperation("view")
                    }}>查看</a>
                    <span style={{ color: "#999", marginTop: "-1px" }}> | </span>
                    <a key="edit"
                        onClick={() => {
                            console.log("record", record)
                            setOperation("edit")
                            setIsadd(true)
                            itemView(record)
                        }}
                    >编辑</a>
                    <span style={{ color: "#999", marginTop: "-1px" }}> | </span>
                    <a key="copy"
                        onClick={() => {
                            setOperation("copy")
                            setIsadd(true)
                            itemView({
                                ...record,
                                graph_name: record.graph_name + "_copy"
                            })
                        }}>复制</a>
                    <span style={{ color: "#999", marginTop: "-1px" }}> | </span>
                    <a key="v" onClick={() => {
                        historyNode(record)
                    }}>历史版本</a>
                    <span style={{ color: "#999", marginTop: "-1px" }}> | </span>
                    <a key="gun" onClick={()=>{
                        rollback_graph(record)
                        queryList({})
                    }}>回滚</a>
                </Space>
            ),
        },
    ];


    const queryList = (params: any) => {
        const { admin, scene_name, status, time, graph_name } = params
        const data = {
            graph_name: graph_name || "",
            status: status || 0,
            scene_name: scene_name || "",
            admin: admin || "",
            min_updated_time: time && time[0] ? time[0] : 0,
            max_updated_time: time && time[1] ? time[1] : 0,
        }
        api.query_graphs({ opr: "query_graphs", data }).then(res => {
            console.log(res)
            if (res.status === 0 && res.data.results) {
                setTableData(res.data.results)
            }
        })
    }
    const newAdd = () => {
        setOperation('new')
        setIsadd(true)
        itemView({})
        setEditStatus(true)
    }

    useEffect(() => {

        console.log(history)
        if(history.location.state) {
            setIsadd(false)
            itemView(history.location.state)
            setOperation("view")
        }
        queryList({})
        api.get_graph_template_names({ opr: "get_graph_template_names" }).then((res: any) => {
            console.log("res.data.template_names", res.data.template_names)
            if (res.status === 0 && res.data.template_names) {
                setTemplate_names(res.data.template_names)
            }
        })
        api.get_scene_names({ opr: "get_scene_names" }).then((res: any) => {
            // console.log(res)
            if (res.status === 0 && res.data.scene_names) {
                setScene_names(res.data.scene_names)
            }
        })
    }, [])




    return (
        <div style={{ padding: "30px 40px" }}>
            {
                !showEdit ? <div>
                    <Card title="查询条件">
                        <QueryForm formRolue={formRolue} form={record} queryList={queryList}></QueryForm>
                    </Card>
                    <Card title="查询结果" extra={<Button type="primary" onClick={() => newAdd()}>新增</Button>} style={{ marginTop: "20px" }}>
                        <Table key="bussiness_id" columns={columns} dataSource={tableData} />
                    </Card>

                </div>
                    :
                    <div>
                        <StepsForm queryList={queryList} operation={operation} formRolue={formRolue} isadd={isadd} form={record} scene_names={scene_names} template_names={template_names} showEdit={showEdit} setEditStatus={setEditStatus}></StepsForm>
                    </div>
            }

            <Modal width={"80%"} title="历史版本" visible={showChart} onCancel={() => { setShowChart(false) }} footer={[
                <Button key="back" onClick={() => {
                    setShowChart(false)
                }}>关闭</Button>
            ]}>
                <Table key="struct_name" columns={nodeColumns1} dataSource={chartTableData} />
            </Modal>
        </div>
    );
}
