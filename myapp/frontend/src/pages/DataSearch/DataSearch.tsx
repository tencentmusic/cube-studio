import Icon, { MenuOutlined, ReloadOutlined, SaveOutlined, StopOutlined } from '@ant-design/icons';
import { Button, message, Modal, Select, Tabs, Tooltip } from 'antd'
import React, { useEffect, useRef, useState } from 'react'
import { actionGetDataSearchRes, actionRun, TTaskStatus } from '../../api/dataSearchApi';
import CodeEdit from '../../components/CodeEdit'
import { IOptionsGroupItem } from '../../components/DataDiscoverySearch/DataDiscoverySearch';
import InputSearch from '../../components/InputSearch/InputSearch';
import TableBox from '../../components/TableBox/TableBox';
import Draggable from 'react-draggable';
import './DataSearch.less';

interface IDataSearchItem {
    tabId: string
    title: string
    appGroup: string
    status: TTaskStatus
    sqlContent?: string
    sqlContentTemporary?: string
    downloadUrl?: string
    reqId?: string
    timer?: any
}
interface IDataSearchItemParams {
    tabId?: string
    title?: string
    appGroup?: string
    status?: TTaskStatus
    sqlContent?: string
    sqlContentTemporary?: string
    downloadUrl?: string
    reqId?: string
    timer?: any
}

type IDataSearchStore = {
    [tabId: string]: IDataSearchItem
}

export default function DataSearch() {
    // const [loadingSearch, setLoadingSearch] = useState(false)
    const [errorMsg, setErrorMsg] = useState<String>()
    const dataByCache = JSON.parse(localStorage.getItem('dataSearch') || JSON.stringify({
        '1': {
            tabId: '1',
            title: '新查询 1',
            status: 'init'
        }
    }))
    const [columnConfig, setColumnConfig] = useState<any[]>([])
    const [dataList, setDataList] = useState<any[]>([])
    const [appGroup, setAppGroup] = useState<string>('')
    const [resLog, setResLog] = useState<string>()
    const showBoxRef: any = useRef()

    // const [runTimer, _setRunTimer] = useState<any>()
    // const runTimerRef = useRef(runTimer);
    // const setRunTimer = (data: any): void => {
    //     runTimerRef.current = data;
    //     _setRunTimer(data);
    // };

    const [inputContent, _setInputContent] = useState<IDataSearchStore>(dataByCache)
    const inputContentRef = useRef(inputContent);
    const setInputContent = (data: IDataSearchStore): void => {
        inputContentRef.current = data;
        _setInputContent(data);
    };

    const [activeKey, _setActiveKey] = useState<string>('1')
    const activeKeyRef = useRef(activeKey);
    const setActiveKey = (data: string): void => {
        activeKeyRef.current = data;
        _setActiveKey(data);
    };

    const initialPanes = Object.entries(inputContent).reduce((pre: IDataSearchItem[], [key, value]) => ([...pre, { ...value }]), [])
    const [panes, setPanes] = useState(initialPanes);
    const newTabIndex = useRef(Object.entries(inputContent).length);

    const setTabState = (currentState: IDataSearchItemParams, key?: string) => {
        const targetRes: IDataSearchStore = {
            ...inputContentRef.current
        }
        if (key !== undefined) {
            targetRes[key] = {
                tabId: currentState.tabId || '',
                title: currentState.title || '',
                status: currentState.status || 'init',
                appGroup,
                ...currentState
            }
        } else {
            targetRes[activeKey] = {
                ...targetRes[activeKey],
                ...currentState
            }
        }

        localStorage.setItem('dataSearch', JSON.stringify(targetRes))
        inputContentRef.current = targetRes
        setInputContent(targetRes)
    }

    const handleData = (result: (string | number)[][]) => {
        const header = result[0] || []
        const data = result.slice(1)
        const targetData = data.map((row) => {
            const rowItem = row.reduce((pre, next, index) => ({ ...pre, [header[index]]: next }), {})
            return rowItem
        })
        const headerConfig = header.map(item => ({
            title: item,
            dataIndex: item,
            key: item,
            width: 100,
        }))
        // console.log('headerConfig', headerConfig);
        // console.log(targetData.slice(0, 10));
        setColumnConfig(headerConfig)
        setDataList(targetData.slice(0, 10))
    }

    useEffect(() => {
        const targetDom = document.getElementById("buttonDrag")
        if (targetDom) {
            drag(targetDom);
        }

        function drag(obj: any) {
            obj.onmousedown = function (e: any) {
                var dir = "";  //设置好方向
                var firstX = e.clientX;  //获取第一次点击的横坐标
                var firstY = e.clientY;   //获取第一次点击的纵坐标
                var width = obj.offsetWidth;  //获取到元素的宽度
                var height = obj.offsetHeight;  //获取到元素的高度
                var Left = obj.offsetLeft;   //获取到距离左边的距离
                var Top = obj.offsetTop;   //获取到距离上边的距离
                //下一步判断方向距离左边的距离+元素的宽度减去自己设定的宽度，只要点击的时候大于在这个区间，他就算右边
                if (firstX > Left + width - 30) {
                    dir = "right";
                } else if (firstX < Left + 30) {
                    dir = "left";
                }
                if (firstY > Top + height - 30) {
                    dir = "down";
                } else if (firstY < Top + 30) {
                    dir = "top";
                }
                //判断方向结束
                document.onmousemove = function (e) {
                    switch (dir) {
                        case "right":
                            obj.style["width"] = width + (e.clientX - firstX) + "px";
                            break;
                        case "left":
                            obj.style["width"] = width - (e.clientX - firstX) + "px";
                            obj.style["left"] = Left + (e.clientX - firstX) + "px";
                            break;
                        case "top":
                            obj.style["height"] = height - (e.clientY - firstY) + "px";
                            obj.style["top"] = Top + (e.clientY - firstY) + "px";
                            break;
                        case "down":
                            obj.style["height"] = height + (e.clientY - firstY) + "px";
                            break;
                    }
                }
                obj.onmouseup = function () {
                    document.onmousemove = null;
                }
                return false;
            }
        }
    }, [])

    // 清空定时器
    useEffect(() => {
        return () => {
            Object.entries(inputContent).forEach((item) => {
                const [key] = item
                clearInterval(inputContentRef.current[key].timer)
            })
        }
    }, [])

    useEffect(() => {
        console.log('activeKey', activeKey);
        // 当前tab状态是runing，触发轮询
        if (inputContent[activeKey] && inputContent[activeKey].status === 'running') {
            pollGetRes(inputContent[activeKey].reqId || '')
        }
        // if (inputContent[activeKey] && inputContent[activeKey].reqId && (inputContent[activeKey].status === 'success' || inputContent[activeKey].status === 'failure')) {
        //     setTabState({ status: 'running' })
        //     actionGetDataSearchRes(inputContent[activeKey].reqId || '').then(res => {
        //         const { result, err_msg, state, result_url } = res.data
        //         if (err_msg) {
        //             setErrorMsg(err_msg)
        //         }
        //         if (state === 'failure') {
        //             setTabState({ status: state })
        //             message.error('查询结果失败，尝试重新运行')
        //         }
        //         if (state === 'success') {
        //             setTabState({ status: state, downloadUrl: result_url })
        //             console.log('result', result);
        //             handleData(result)
        //         }
        //     }).catch(() => {
        //         setTabState({ status: 'failure' })
        //     })
        // }
    }, [activeKey])


    const onChange = (newActiveKey: string) => {
        Object.entries(inputContent).forEach((item) => {
            const [key] = item
            if (key !== newActiveKey) {
                clearInterval(inputContent[key].timer)
            }
        })
        setErrorMsg(undefined)
        setResLog(undefined)
        setColumnConfig([])
        setDataList([])
        setActiveKey(newActiveKey);
    };

    const add = () => {
        clearInterval(inputContentRef.current[activeKey].timer)

        const uniqueKey = Math.random().toString(36).substring(2);
        const newActiveKey = uniqueKey;
        const title = `新查询 ${++newTabIndex.current}`
        const newPanes = [...panes];
        newPanes.push({ title, tabId: newActiveKey, status: 'init', appGroup });
        setPanes(newPanes);
        setActiveKey(newActiveKey);

        let res: IDataSearchStore = {
            ...inputContent, [newActiveKey]: {
                tabId: newActiveKey,
                title,
                status: 'init',
                appGroup,
            }
        }
        setInputContent(res)
        localStorage.setItem('dataSearch', JSON.stringify(res))
    };

    const remove = (targetKey: string) => {
        let newActiveKey = activeKey;
        let lastIndex = -1;
        panes.forEach((pane, i) => {
            if (pane.tabId === targetKey) {
                lastIndex = i - 1;
            }
        });
        const newPanes = panes.filter(pane => pane.tabId !== targetKey);
        if (newPanes.length && newActiveKey === targetKey) {
            if (lastIndex >= 0) {
                newActiveKey = newPanes[lastIndex].tabId;
            } else {
                newActiveKey = newPanes[0].tabId;
            }
        }
        setPanes(newPanes);
        setActiveKey(newActiveKey);

        let res = { ...inputContent }
        delete res[targetKey]
        setInputContent(res)
        localStorage.setItem('dataSearch', JSON.stringify(res))
    };

    const onEdit = (targetKey: any, action: 'add' | 'remove') => {
        if (action === 'add') {
            add();
        } else {
            remove(targetKey);
        }
    };

    const fetchData = (task_id: string) => {
        actionGetDataSearchRes(task_id).then(res => {
            console.log(res.data)
            const { state, result, err_msg, result_url, spark_log_url } = res.data
            if (state === 'running') {
                setResLog(spark_log_url)
            }
            if (state === 'success') {
                setTabState({ status: state, downloadUrl: result_url })
                clearInterval(inputContentRef.current[activeKey].timer)
                handleData(result)
            }
            if (state === 'failure') {
                setTabState({ status: state })
                clearInterval(inputContentRef.current[activeKey].timer)
                setErrorMsg(err_msg)
            }
        }).catch(() => {
            clearInterval(inputContentRef.current[activeKey].timer)
            message.error('查询结果失败，尝试重新运行')
            setTabState({ status: 'failure' })
        })
    }

    const pollGetRes = (task_id: string) => {
        if (inputContentRef.current[activeKey].timer) {
            clearInterval(inputContentRef.current[activeKey].timer)
        }
        let timer = setInterval(() => {
            fetchData(task_id)
        }, 10000)
        setTabState({ reqId: task_id, status: 'running', timer })

        fetchData(task_id)
    }

    return (
        <div className="datasearch-container p16 fade-in">
            <Tabs type="editable-card" onChange={onChange} activeKey={activeKey} onEdit={onEdit}>
                {panes.map((pane, index) => (
                    <Tabs.TabPane tab={`新查询 ${index + 1}`} key={pane.tabId} closable={index !== 0}>
                        <div className="d-f fd-c h100">
                            <div className="flex2 s0 ov-a">
                                <CodeEdit
                                    value={inputContent[activeKey]?.sqlContent}
                                    onSelect={(value) => {

                                    }}
                                    onChange={(value) => {
                                        const res: IDataSearchStore = {
                                            ...inputContent,
                                            [activeKey]: {
                                                tabId: activeKey,
                                                sqlContent: value === '' ? undefined : value,
                                                title: pane.title,
                                                status: inputContent[activeKey]?.status,
                                                appGroup: inputContent[activeKey]?.appGroup
                                            }
                                        }
                                        localStorage.setItem('dataSearch', JSON.stringify(res))
                                        setInputContent(res)
                                    }} />
                            </div>

                            <div className="ov-a" id="showBox" ref={showBoxRef} style={{ height: 500 }}>
                                <Draggable
                                    axis="y"
                                    onStart={() => {}}
                                    onDrag={(e: any) => {
                                        const showBoxDom = document.getElementById('showBox')
                                        if (showBoxDom) {
                                            const res = document.body.clientHeight - e.y
                                            showBoxDom.style.height = `${res}px`
                                        }
                                    }}
                                    onStop={() => {}}>
                                    <div className="ta-c" style={{ cursor: 'ns-resize' }}><MenuOutlined /></div>
                                </Draggable>
                                <div className="p8 bor-l bor-r b-side d-f ac jc-r bg-w">
                                    {/* <Tooltip title="jupyter">
                                        <img className="mr16 cp" src={require('../../images/jupyter.svg').default} style={{ width: 20 }} alt="" onClick={() => {
                                            window.open(`${window.location.protocol}//${window.location.hostname}/idex/`, 'bank')
                                        }} />
                                    </Tooltip> */}
                                    <Tooltip title="保存">
                                        <SaveOutlined className="mr16 cp" style={{ fontSize: 18 }} onClick={() => {
                                            localStorage.setItem('dataSearch', JSON.stringify(inputContent))
                                            message.success('保存成功')
                                        }} />
                                    </Tooltip>
                                    {/* <Tooltip title="刷新">
                                    <ReloadOutlined className="mr16 cp" style={{ fontSize: 18 }} />
                                </Tooltip> */}
                                    <InputSearch
                                        value={inputContent[activeKey].appGroup}
                                        isOpenSearchMatch
                                        onChange={(value) => {
                                            setTabState({ appGroup: value })
                                            // setAppGroup(value)
                                        }} options={['队里1',
                                            '队里2',
                                            '队里3',
                                            '队里4'
                                        ]} placeholder="应用组" width={'400px'} />
                                    <Button
                                        type='default'
                                        className="mlr16"
                                        disabled={inputContent[activeKey].status !== 'success'}
                                        onClick={() => {
                                            window.open(inputContent[activeKey].downloadUrl, 'bank')
                                        }}
                                    >下载结果</Button>
                                    <Button type="default" disabled={inputContent[activeKey].status !== 'running'} className="mr16" onClick={() => {
                                        Modal.confirm({
                                            title: '终止',
                                            icon: <StopOutlined />,
                                            content: '确定终止?',
                                            okText: '确认终止',
                                            cancelText: '取消',
                                            okButtonProps: { danger: true },
                                            onOk() {
                                                return new Promise((resolve, reject) => {

                                                    setTabState({
                                                        status: 'init',
                                                        reqId: undefined
                                                    })
                                                    clearInterval(inputContentRef.current[activeKey].timer)
                                                    resolve('');
                                                })
                                                    .then((res) => {
                                                        message.success('终止成功');
                                                    })
                                                    .catch(() => {
                                                        message.error('终止失败');
                                                    });
                                            },
                                            onCancel() { },
                                        });
                                    }}><StopOutlined /> 终止</Button>
                                    <Button type="primary" loading={inputContent[activeKey].status === 'running'} onClick={() => {
                                        if (inputContent[activeKey].appGroup) {
                                            setTabState({ status: 'running' })
                                            setErrorMsg(undefined)
                                            setResLog(undefined)

                                            actionRun({
                                                tdw_app_group: inputContent[activeKey].appGroup,
                                                sql: inputContent[activeKey]?.sqlContent || ''
                                            }).then(res => {
                                                console.log('task_id', res.data.task_id)
                                                const { err_msg, task_id } = res.data
                                                if (err_msg) {
                                                    setTabState({ status: 'failure' })
                                                    setErrorMsg(err_msg)
                                                }
                                                if (task_id) {
                                                    pollGetRes(task_id)
                                                }
                                            }).catch(err => {
                                                setTabState({ status: 'failure' })
                                            })
                                        } else {
                                            message.warning('请先选择应用组')
                                        }
                                    }}>运行</Button>
                                </div>
                                <div className="flex1 bor b-side s0 ov-a bg-w p-r h100">
                                    <div className="pt8">
                                        <div className="tag-result bg-theme c-text-w mr16">
                                            结果
                                    </div>
                                        {
                                            resLog ? <Button type='link' size="small" onClick={() => {
                                                window.open(resLog, 'bank')
                                            }}>查看日志</Button> : null
                                        }

                                    </div>
                                    <div className="plr16 pt8">
                                        {
                                            errorMsg ? errorMsg : <TableBox
                                                loading={inputContent[activeKey].status === 'running'}
                                                cancelExportData={true}
                                                rowKey={(record: any) => {
                                                    return JSON.stringify(record)
                                                }}
                                                columns={columnConfig}
                                                pagination={false}
                                                dataSource={dataList}
                                                scroll={{ x: 1200 }}
                                            />
                                        }
                                    </div>
                                </div>
                            </div>
                        </div>
                    </Tabs.TabPane>
                ))}
            </Tabs>
        </div>

    )
}
