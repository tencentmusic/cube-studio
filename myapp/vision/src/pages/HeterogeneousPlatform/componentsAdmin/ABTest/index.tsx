import React, { useState, useEffect } from 'react';
import {
    Card,
    Button,
    Table, Space,
    Modal,
    message,
    Tag
} from 'antd';
import QueryForm from "../components/QueryForm/index"
import ABTestEdit from "../components/ABTestEdit/index"
import api from "@src/api/admin"
import { formRolue, form } from "./data.js"
import { STATUS, STATUS_COLOR } from "../common"
import {useHistory} from "react-router-dom"
import './index.css';

export default function ABTest() {
    const [showEdit, setEditStatus] = useState(false)
    const [tableData, setTableData] = useState([])
    const [chartTableData, setChartTableData] = useState([])
    const [sceneArr, setSceneArr] = useState([])
    const [chartArr, setChartArr] = useState({graphArr: [], arr:[]})
    const [viewForm, setViewForm] = useState<any>({})
    const [isadd, setIsadd] = useState(false)
    const [showChart, setShowChart] = useState(false)
     

    const history = useHistory()

    const relationNode = (record: any) => {
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


    const newAdd = () => {
        setIsadd(true)
        setViewForm({ name: "" })
        setEditStatus(true)
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
                    <a key="view" onClick={() => {
                        history.push({pathname: "/HeterogeneousPlatform/componentsAdmin/chartAdmin", state: {...record} })
                    }}>查看</a>
                    {/* <span style={{color: "#999"}}>
                        查看
                    </span> */}
                </Space>
            ),
        },
    ]




    const columns = [
        {
            title: '#',
            render: (text: any, record: any, index: number) => <a>{index + 1 + ''}</a>,
        },
        {
            title: 'ABTest名',
            fType: "input",
            dataIndex: 'abtest_name',
            key: 'abtest_name',
        },
        {
            title: '业务id',
            fType: "input",
            dataIndex: 'bussiness_id',
            key: 'bussiness_id',
        },
        {
            title: '频道id',
            fType: "input",
            dataIndex: 'channel_id',
            key: '"channel_id',
        },
        {
            title: '模块id',
            fType: "textArea",
            dataIndex: 'module_id',
            key: 'module_id',
        },
        {
            title: '客户端版本',
            fType: "textArea",
            dataIndex: 'client_version',
            key: 'client_version',
        },
        {
            title: '负责人',
            fType: "textArea",
            dataIndex: 'admin',
            key: 'admin',
        },
        {
            title: '状态',
            fType: "textArea",
            dataIndex: 'status',
            key: 'status',
            render: (text: any) => <Tag color={STATUS_COLOR[text] ? STATUS_COLOR[text] : ""}>{ STATUS[text] ? STATUS[text] : ""}</Tag> 
        },
        {
            title: '场景',
            fType: "textArea",
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
                        console.log(record)
                        const arr = []
                        if (record.config) {
                            const config = JSON.parse(record.config)
                            for (const key in config) {
                                arr.push({
                                    param_name: config[key].param_name,
                                    param_value: config[key].param_value,
                                    graph_name: config[key].graph_name,
                                    version: config[key].version
                                })
                            }
                        }
                        const data = {
                            abtest_name: record.abtest_name,
                            bussiness_id: record.bussiness_id,
                            channel_id: record.channel_id,
                            module_id: record.module_id,
                            client_version: record.client_version,
                            status: record.status,
                            admin: record.admin,
                            ...record,
                            config: arr,
                        }
                        setIsadd(false)
                        setViewForm(data)
                        setEditStatus(true)
                    }}>查看</a>
                    <span style={{ color: "#999", marginTop: "-1px" }}> | </span>
                    <a key="edit" onClick={() => {
                        const arr = []
                        if (record.config) {
                            const config = JSON.parse(record.config)
                            for (const key in config) {
                                arr.push({
                                    param_name: config[key].param_name,
                                    param_value: config[key].param_value,
                                    graph_name: config[key].graph_name,
                                    version: config[key].version
                                })
                            }
                        }
                        const data = {
                            abtest_name: record.abtest_name,
                            bussiness_id: record.bussiness_id,
                            channel_id: record.channel_id,
                            module_id: record.module_id,
                            client_version: record.client_version,
                            status: record.status,
                            admin: record.admin,
                            version: record.version,
                            ...record,
                            config: arr,
                        }
                        console.log(record)
                        setIsadd(true)
                        setViewForm(data)
                        setEditStatus(true)

                    }}>编辑</a>
                    <span style={{ color: "#999", marginTop: "-1px" }}> | </span>
                    <a key="node" onClick={() => { relationNode(record) }}> 关联图 </a>
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
        console.log(params)
        const { admin, abtest_name, channel_id, time, bussiness_id, module_id, client_version, scene_name } = params
        const data = {
            abtest_name: abtest_name || "",
            bussiness_id: bussiness_id || "", // 负责人
            channel_id: channel_id || "",
            module_id: module_id || "",
            client_version: client_version || "",
            status: status || 0,
            scene_name: scene_name || "",
            admin: admin || "",
            min_updated_time: time && time[0] ? new Date(time[0]).getTime() : 0,
            max_updated_time: time && time[1] ? new Date(time[1]).getTime() : 0,
        }
        api.query_abtests({ opr: "query_abtests", data }).then(res => {
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
        if (viewForm.version !== undefined) {
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
        api.get_scene_names({ opr: "get_scene_names" }).then(res => {
            if (res.status === 0) {
                setSceneArr(res.data.scene_names)
            } else {
                message.error(res.message);
            }
        })
    }, [])


    const scene_nameChange = (e: any) => {
        api.get_scene_available_graphs({ opr: "get_scene_available_graphs", data: { scene_name: e } }).then(res => {
            if (res.status === 0) {
                const data:any  = {
                    graphArr: Object.keys(res.data.results),
                    arr: res.data.results
                }
                setChartArr(data)
            } else {
                setChartArr({graphArr: [], arr: []})
                message.error(res.message);
            }
        })
    }

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
                <Card title="查询结果" extra={<Button type="primary" onClick={() => newAdd()}>新增</Button>} style={{marginTop: "20px"}}>
                    <Table key="bussiness_id" columns={columns} dataSource={tableData} />
                </Card>

            </div>
            <Modal width={900} title={isadd ? isadd && viewForm.version ? "编辑" : "新增" : "查看"} afterClose={()=> {setViewForm({})}} visible={showEdit} onCancel={() => { setEditStatus(false) }} footer={[
                <Button key="back" onClick={() => {
                    setEditStatus(false)
                }}>关闭</Button>
            ]}>
                <ABTestEdit scene_nameChange={scene_nameChange} formRolue={viewFormRolue} form={viewForm} chartArr={chartArr} isadd={isadd} showEdit={showEdit} sceneArr={sceneArr} submitForm={submitForm} resFied={resFied} />
            </Modal>

            <Modal width={"60%"} title="关联图" visible={showChart}  onCancel={() => { setShowChart(false) }} footer={[
                <Button key="back" onClick={() => {
                    setShowChart(false)
                }}>关闭</Button>
            ]}>
                <Table key="struct_name" columns={nodeColumns} dataSource={chartTableData} />
            </Modal>

        </div>
    );
}
