import React, { useState, useEffect, useRef } from 'react';
import { render } from 'react-dom';
import { Steps, Button, Menu, Row, Col, Collapse, message, Tooltip } from 'antd';

import ChartG6 from "../ChartG6/index"
import EditForm from "../EditForm/index"
import ChartNodeEdit from "../ChartNodeEdit/index"
import ChartEdgeEdit from "../ChartEdgeEdit/index"
import ReactJson from 'react-json-view';
import "./index.css"
import api from "@src/api/admin"

import G6 from '@antv/g6';

const { SubMenu } = Menu;
const { Step } = Steps;
const { Panel } = Collapse;
let delItem:any = []
export default function StepsForm(prop: any) {
    const [chartData, setChartData] = useState<any>({ nodes: [], edges: [] })
    const [stepCurrent, setCurrent] = useState(0)
    const [viewObj, SetObj] = useState<any>({})
    const [graph, SetGraph] = useState<any>(null)
    const [graphData, SetGraphData] = useState<any>({})
    const [baseForm, setBaseForm] = useState<any>({})
    const [activeKey, setActiveKey] = useState<any>([])
    const [nodeId, setNodeId] = useState("")
    const [menuArr, steMenuArr] = useState([])
    const [nodeForm, setNodeForm] = useState<any>({})
    const chartRef = useRef(null)
    const [nodeInfoArr, setNodeInfoArr] = useState([])
    const [edgeInfoArr, setedGeInfoArr] = useState([])
    const [edgeName, setEdgeName] = useState<any>({})
    const [map_config, setMap_config] = useState<any>({})
    const [newMap_config, SetNewMap_config] = useState<any>({})
    
    const [chartsData, setChartsData] = useState<any>({})
    const [saveFrom, setSAaveFrom] = useState<any>({})


    // const [delItem, setDelItem] = useState<any>([])

    useEffect(() => {
        setBaseForm(prop.form)
    }, [prop.form])
    useEffect(()=>{
        delItem =[]
    }, [stepCurrent])

    const callback = (key: any) => {
        setActiveKey(key)
    }
    const fittingString = (str: any, maxWidth: any, fontSize: any) => {
        const ellipsis = '...';
        const ellipsisLength = G6.Util.getTextSize(ellipsis, fontSize)[0];
        let currentWidth = 0;
        let res = str;
        const pattern = new RegExp('[\u4E00-\u9FA5]+'); // distinguish the Chinese charactors and letters
        str.split('').forEach((letter: any, i: any) => {
            if (currentWidth > maxWidth - ellipsisLength) return;
            if (pattern.test(letter)) {
                currentWidth += fontSize;
            } else {
                // get the width of single letter according to the fontSize
                currentWidth += G6.Util.getLetterWidth(letter, fontSize);
            }
            if (currentWidth > maxWidth - ellipsisLength) {
                res = `${str.substr(0, i)}${ellipsis}`;
            }
        });
        return res;
    };
    let arr: any = []
    const pushDelItem = (item: any) =>{
       
        // cccc.push(item)
        // console.log([...cccc, item])
        // setDelItem([item])
        delItem=[...delItem, item]
    }

    const nodeSubmit = (value: any) => {
        const { config, des, factory_name, name, type, admin, level } = value.value
        const n = graph.find("node", (node: any) => node._cfg.id === value.form.id)
        graph.updateItem(n, { config, des, factory_name, name, atype: type, label: name, dataType: name, admin, level, modify: true, isAdd: false })
        message.success("更新成功")
    }
    const edgeSubmit = (value: any, op: any) => {
        console.log(value, op)
        const { config, des, factory_name, name, admin } = value
        const n = graph.find("edge", (node: any) => node._cfg.id === op.id)
        graph.updateItem(n, { config, des, factory_name, name, admin, isEdit: true, modify: true })
    }

    const submitForm = (form: any, name: any) => {
        if (name === 'form') {
            setBaseForm({
                ...form,
                version: prop.form.version
            })
             

            setSAaveFrom({ ...form,
                version: prop.form.version})
         
            api.batch_query_components({ opr: "batch_query_components", data: { graph_name: form.graph_name, version: prop.form.version } }).then(res => {
                console.log("res, version", res)
                if (res.status === 0 && res.data.results) {
                    setNodeInfoArr(res.data.results)
                }
            })

            console.log("map_config12312", map_config)
            if (prop.form.map_config) {
                const map_config = JSON.parse(prop.form.map_config)
                setMap_config(map_config)
                console.log("map_config", map_config)
                const nodes: any = []
                const edges: any = []
                for (const key in map_config) {
                    // let nodeItem: any = nodeInfoArr.find((A: any) => A.name === key)
                    // console.log(nodeItem)
                    // if (!nodeItem) nodeItem = {}
                    nodes.push({
                        id: map_config[key].id + "",
                        label: fittingString(key, 120, 12),
                        dataType: key,
                        factory: map_config[key].level.factory,
                        level: map_config[key].level,
                        atype: map_config[key].type,
                        // ...nodeItem,
                        isEdit: true
                    })
                    if (map_config[key].edges) {
                        map_config[key].edges.forEach((item: any) => {
                            let edgeItem: any = nodeInfoArr.find((A: any) => A.name === item.name)
                            if (!edgeItem) edgeItem = {}
                            edges.push({
                                source: map_config[item.from].id + "",
                                target: map_config[key].id + "",
                                edgeName: item.name,
                                factory_name: edgeItem ? edgeItem.name : "",
                                ...edgeItem
                            })
                        })

                    }
                }
                setChartData({ nodes, edges })
            } else {
                setChartData({
                    nodes: [
                        {
                            id: "-2",
                            label: "end",
                            dataType: "end",
                        },
                        {
                            id: "-1",
                            label: "start",
                            dataType: "start",
                        }
                    ], edges: []
                })
            }

            api.query_template_available_node_factorys({ opr: "query_template_available_node_factorys", data: { template_name: form.template_name } }).then(res => {
                if (res.status === 0 && res.data.results) {
                    steMenuArr(res.data.results)
                }
            })


            api.get_edge_factory_names({ opr: "get_edge_factory_names" }).then(res => {
                console.log("res, version", res)
                if (res.status === 0 && res.data.edge_factory_names) {
                    setedGeInfoArr(res.data.edge_factory_names)
                }
            })

            arr = setCurrent(stepCurrent + 1)
        }
    }
    console.log("prop.scene_names", prop.scene_names)
    const baseFormRolue = [
        {
            type: 'input',
            name: 'graph_name',
            label: '图名称'
        },
        {
            type: 'input',
            name: 'admin',
            label: '负责人'
        },
        {
            type: 'select',
            name: 'scene_name',
            label: '使用场景',
            options: prop.scene_names ? prop.scene_names.map((item: any) => {
                return {
                    label: item,
                    value: item
                }
            }) : []
        },
        {
            type: 'select',
            name: 'template_name',
            label: '图模板',
            options: prop.template_names ? prop.template_names.map((item: any) => {
                return {
                    label: item,
                    value: item
                }
            }) : []
        },
        {
            type: 'radio',
            name: 'status',
            label: '环境',
            options: [
                {
                    label: "测试",
                    value: 2
                },
                {
                    label: "正式",
                    value: 3
                },
                {
                    label: "只创建",
                    value: 1
                }
            ]
        },
        {
            type: 'textArea',
            name: 'des',
            label: '描述'
        },
    ]


    const menuItemDragEnd = (e: any, type: any) => {
        console.log(type)
        const current: any = chartRef && chartRef.current ? chartRef.current : {}
        if (current.offsetLeft) {
            if (e.screenX > current.offsetLeft && e.screenX < current.offsetLeft + current.offsetWidth && e.screenY > current.offsetTop && e.screenY < current.offsetTop + current.offsetHeight && graph) {
                const idArr = graph.save().nodes.map((item: any) => Number.parseInt(item.id))
                console.log( e.target.childNodes[0].innerText,)
                graph.addItem('node', {
                    id: (Math.max(...idArr) + 1) + "",
                    label: e.target.childNodes[0].innerText,
                    // name:e.target.childNodes[0].innerText,
                    typeName: type,
                    dataType: e.target.childNodes[0].innerText,
                    x: e.screenX - current.offsetLeft / graph.getZoom().toFixed(2),
                    y: e.screenY - current.offsetTop / graph.getZoom().toFixed(2),
                    isAdd: true
                }, false)
            }

        }
    }

    const getGraph = (graph1: any) => {
        SetGraph(graph1)
        graph1.on('node:click', (ev: any) => {
            console.log(ev.item._cfg.model)
            if(ev.item._cfg.model.label === "start" ||ev.item._cfg.model.label === "end" ) {
                setActiveKey([])
                return false
            }
            setActiveKey(["1"])
            setNodeForm({ ...ev.item._cfg.model, name: ev.item._cfg.model.dataType, typeName: ev.item._cfg.model.typeName })
            setEdgeName({})
            setNodeId(ev.item._cfg.model.id)
        });
        graph1.on('edge:mouseenter', (evt: any) => {
            const { item } = evt;
            graph1.setItemState(item, 'active', true);
        });

        graph1.on('edge:mouseleave', (evt: any) => {
            const { item } = evt;
            graph1.setItemState(item, 'active', false);
        });

        graph1.on('edge:click', (evt: any) => {
            console.log("evt", evt)
            setNodeForm({})
            setNodeId("0")
            setActiveKey(["2"])
            setEdgeName({ ...evt.item._cfg.model, edgeName: evt.item._cfg.model.edgeName ? evt.item._cfg.model.edgeName : "", })
            const { item } = evt;
            graph1.getEdges().forEach((edge:any) => {
                graph1.setItemState(edge, 'selected', false);
            })
            graph1.setItemState(item, 'selected', true);
        });
        graph1.on('canvas:click', (evt: any) => {
            graph1.getEdges().forEach((edge: any) => {
                graph1.clearItemStates(edge);
            });
        });



    }
    const getGraphData = (graphData1: any) => {
        SetGraphData(graphData1)
    }

    const stepSubmit2 = () => {
        const data = graph.save()
        setChartsData(data)
        console.log("dat", data)
        let isCheck = true
        console.log(data)
        data.nodes.forEach((item: any) => {
            const index = data.edges.find((o: any) => item.id === o.source || item.id === o.target)
            if (index === undefined) {
                isCheck = false
            }
        })
        if (!isCheck) {
            message.error("请填写完整信息")
            return false
        }
        const config: any = []
        const obj: any = {}
        const itemListObj: any = []
        let isOK:any = true
        data.nodes.forEach((item: any) => {
            console.log(item)
            const nodeItem = graph.find('node', (node: any) => {
                return node.get('model').id === item.id;
            })
            const m: any = nodeInfoArr.find((T: any) => T.name === item.dataType)
            obj[item.dataType] = {
                factory: item.dataType,
                id: item.id,
                level: item.level ? item.level : 0,
                type: item.atype ? item.atype : m ? m.type : "",
                edges: nodeItem ? nodeItem.getInEdges().map((O: any, index: any) => {
                    return {
                        factory: O._cfg.model.factory_name ? O._cfg.model.factory_name : "",
                        from: O.getSource()._cfg.model.dataType,
                        name: O._cfg.model.edgeName || "EmptyEdge" + index,
                    }
                }) : []
            }
            console.log(m)
            if (item.isAdd) {
                isOK = false
                // config.push({
                //     type: item.atype ? item.atype : "",
                //     factory_name: item.dataType ? item.dataType : "",
                //     name: item.factory_name ? item.factory_name : "",
                //     des: item.des ? item.des : "",
                //     graph_name: baseForm.graph_name,
                //     admin: item.admin ? item.admin : "",
                //     // level: item.level ? item.level : 0,
                //     config: item.config ? item.config : "",
                //     status: 5,
                //     output_struct: item.output_struct ? item.output_struct : "",
                //     version: 1,
                // })
            }

            if (item.modify) {
                config.push({
                    ...m,
                    type: item.atype ? item.atype : "",
                    factory_name: item.factory_name ? item.factory_name : m.factory_name,
                    name: item.name ? item.name : "",
                    des: item.des ? item.des : "",
                    graph_name: baseForm.graph_name,
                    admin: item.admin ? item.admin :"",
                    config: item.config ? item.config : "",
                    // level: item.level ? item.level : 0,
                    status: item.status ? item.status :5,
                    output_struct: item.output_struct ? item.output_struct : "",
                    version: m ? m.version + 1 : 1,
                })
            }
        })
          console.log("delItem", delItem)
        delItem.forEach((Y: any) => {
            console.log("Y", Y)
            const m: any = nodeInfoArr.find((T: any) => T.name === Y.dataType)
            if(m) {
                config.push({
                    type: m.atype ? m.atype : "",
                    factory_name: m.factory_name ? m.factory_name : "",
                    name: m.name ? m.name : "",
                    des: m.des ? m.des : "",
                    graph_name: baseForm.graph_name,
                    admin: m.admin ? m.admin : "",
                    config: m.config ? m.config : "",
                    // level: item.level ? item.level : 0,
                    status: -1,
                    output_struct: m.output_struct ? m.output_struct : "",
                    version: m.version ? m.version + 1 : 1,
                })
            }   
           
        })

        data.edges.forEach((item: any) => {
            if (item.modify) {
                config.push({
                    admin: item.admin ? item.admin : "",
                    config: item.config ? item.config : "",
                    des: item.des ? item.des : "",
                    factory_name: item.factory_name ? item.factory_name : "",
                    graph_name: baseForm.graph_name,
                    name: item.name ? item.name : "",
                    output_struct: "",
                    status: item.status ? item.status : 0,
                    type: "edge",
                    version: item.version ? item.version + 1 : 1,
                })
            }
        })
        if(isOK === false) {
            message.error("必选字段未填写")
            return false
        }
        SetObj(obj)
        SetNewMap_config(config)
        setCurrent(stepCurrent + 1)
    }

    const completeGraph = () => {
        const data = {
            components_config: newMap_config,
            graph_config: { ...baseForm, version: baseForm.version ? baseForm.version + 1 : 1 ,map_config: JSON.stringify(viewObj)},
           
        }
        if (prop.operation === "new") {
            console.log(prop.operation)
            api.add_graph({ opr: "add_graph", data }).then(res => {
                console.log(res)
                if(res.status === 0) {
                    message.success("添加成功")
                    setCurrent(0)
                    prop.setEditStatus(false)
                    prop.queryList({})
                }else {
                    message.error(res.message)
                }
            })
        } else if (prop.operation === "edit") {
            api.mod_graph({ opr: "mod_graph", data }).then(res => {
                if(res.status === 0) {
                    message.success("修改成功")
                    prop.setEditStatus(false)
                    prop.queryList({})
                    setCurrent(0)
                }else {
                    message.error(res.message)
                }
            })

        }else{
            prop.setEditStatus(false)
            setCurrent(0)
        }
        // prop.setEditStatus(false)
    }

    const menuName: any = {
        recall: "召回节点工厂",
        filter: "过滤节点工厂",
        sort: "排序节点工厂",
        merge: "聚合节点工厂",
        topn: "topN节点工厂",
        custome: "自定义节点工厂",
        preprocess: "预处理节点工厂",
        edge: "边节点工厂",
        async: "异步工厂"
    }


    return (
        <div>
            <Steps current={stepCurrent} style={{ paddingBottom: "20px" }}>
                <Step title="基本信息" />
                <Step title="图配置" />
                <Step title="完成" />
            </Steps>

            {
                stepCurrent === 0 && <div>
                    <EditForm setBaseForm={setBaseForm} opName="form" isadd={prop.isadd} submitForm={submitForm} formRolue={baseFormRolue} saveFrom={saveFrom} form={prop.form} setEditStatus={prop.setEditStatus} stepCurrent={stepCurrent} setCurrent={setCurrent} nodeId={nodeId}></EditForm>
                </div>
            }
            {
                stepCurrent === 1 && <div>
                    <div>
                        <Row gutter={10} style={{ minHeight: "60Vh" }}>
                            <Col span={4}>
                                <Menu
                                    // style={{ width: 256 }}
                                    defaultSelectedKeys={['1']}
                                    defaultOpenKeys={['sub1']}
                                    mode="inline"
                                >
                                    {
                                        menuArr.map((item: any, index: number) => (
                                            <SubMenu key={'sub' + (index + 1)} title={menuName[item.type]}>
                                                {
                                                    item.available_node_factorys ? item.available_node_factorys.map((item1: any, index1: any) => (
                                                        <Tooltip placement="right" title={item1.des} key={index + "-" + index1}>
                                                            <Menu.Item key={(index1 + 1) + (index + 1 + "")} draggable="true" onDragEnd={(e) => {
                                                                if (prop.operation !== 'view') {
                                                                    menuItemDragEnd(e, item.type)
                                                                }
                                                            }}>{item1.factory_name}</Menu.Item>
                                                        </Tooltip>
                                                    )) : null
                                                }
                                            </SubMenu>
                                        ))
                                    }

                                </Menu>
                            </Col>
                            <Col span={14} ref={chartRef} >
                                <ChartG6 chartData={chartData} style={{ height: '100%', background: "#eee" }} chartsData={chartsData} pushDelItem={pushDelItem} getGraph={getGraph} nodeInfoArr={nodeInfoArr} getGraphData={getGraphData}></ChartG6>
                            </Col>
                            <Col span={6}>
                                <Collapse activeKey={activeKey} onChange={callback} accordion collapsible={'disabled'}>
                                    <Panel header="节点配置" key="1">
                                        <ChartNodeEdit operation={prop.operation} baseForm={baseForm} isadd={prop.isadd} nodeSubmit={nodeSubmit} nodeInfoArr={nodeInfoArr} opName="node" form={nodeForm} nodeId={nodeId}></ChartNodeEdit>
                                    </Panel>
                                    <Panel header="边配置" key="2">
                                        <ChartEdgeEdit operation={prop.operation} baseForm={baseForm} isadd={prop.isadd} edgeSubmit={edgeSubmit} edgeInfoArr={edgeInfoArr} nodeInfoArr={nodeInfoArr} opName="egat" form={nodeForm} edgeName={edgeName}></ChartEdgeEdit>
                                    </Panel>
                                </Collapse>
                                <div style={{ display: "flex", justifyContent: 'center', marginTop: "20px" }}>
                                    <Button onClick={() => {
                                        setCurrent(stepCurrent - 1)
                                    }}>上一步</Button>
                                    <Button style={{ marginLeft: "20px" }} onClick={() => {
                                        stepSubmit2()
                                    }}>下一步</Button>
                                </div>
                            </Col>
                        </Row>

                    </div>

                </div>
            }
            {
                stepCurrent === 2 && <div>
                    <Row>
                        <Col span={10} offset={1}>
                            <ReactJson
                                style={{ height: "400px", overflowY: "auto" }}
                                src={viewObj}
                                displayDataTypes={false}
                                theme="railscasts"
                            />
                            <div className="chart-name">图结构配置</div>
                        </Col>
                        <Col span={10} offset={2}>
                            <ReactJson
                                style={{ height: "400px", overflowY: "auto" }}
                                src={newMap_config}
                                displayDataTypes={false}
                                theme="railscasts"
                            />
                            <div className="chart-name">节点或边的私有配置</div>
                        </Col>
                    </Row>
                </div>
            }

            <div style={{ display: "flex", justifyContent: 'center', marginTop: "30px" }}>
                {/* {
                    stepCurrent === 0 ? <Button onClick={() => {
                        prop.setEditStatus(false)
                    }}> 返回 </Button> : null
                } */}


                {
                    stepCurrent === 2 ? <Button style={{ marginLeft: '20px' }} onClick={() => {
                        setCurrent(stepCurrent - 1)
                    }}> 上一步 </Button> : null
                }
                {
                    stepCurrent === 2 ? <Button onClick={() => {
                        completeGraph()
                        setChartsData({})
                        setBaseForm({})
                    }}> 完成 </Button> : null
                }
                {/* {
                    stepCurrent === 0 ? <Button style={{ marginLeft: '20px' }} onClick={() => {
                        setCurrent(stepCurrent + 1)
                    }}> 下一步 </Button> : null
                } */}

            </div>
        </div>
    )
}