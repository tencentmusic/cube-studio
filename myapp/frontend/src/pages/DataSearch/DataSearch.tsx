import Icon, { ExclamationCircleOutlined, MenuOutlined, ReloadOutlined, RightCircleOutlined, SaveOutlined, SlidersOutlined, StarOutlined, StopOutlined } from '@ant-design/icons';
import { Button, Drawer, message, Modal, Select, Switch, Tabs, Tooltip } from 'antd'
import React, { useEffect, useRef, useState } from 'react'
import { actionGetDataSearchRes, actionRun } from '../../api/dataSearchApi';
import CodeEdit from '../../components/CodeEdit'
import InputSearch from '../../components/InputSearch/InputSearch';
import Draggable from 'react-draggable';
import './DataSearch.less';
import TaskList from './TaskList';
import { IEditorTaskItem, IEditorStore, IEditorItem, IEditorItemParams } from './interface';
import moment from 'moment'
import { data2Time } from '../../util';
// import LineChartTemplate from '../../components/LineChartTemplate/LineChartTemplate';
import LoadingStar from '../../components/LoadingStar/LoadingStar';
import cookies from 'js-cookie';
const userName = cookies.get('myapp_username')

const createId = () => {
    return Math.random().toString(36).substring(2)
}

const sqlMap: Record<string, string> = {
    'test': `2023A
Sql Automation
`,
}

export default function DataSearch() {
    const initId = createId()
    const initCurrentEditorData: IEditorItem = {
        tabId: initId,
        title: '新查询 1',
        status: 'init',
        appGroup: '',
        biz: 'your',
        smartShow: false,
        smartContent: '',
        smartCache: '',
        smartTimer: undefined,
        loading: false,
        taskMap: {}
    }
    const initEditorData: Record<string, IEditorItem> = JSON.parse(localStorage.getItem('dataSearch2') || JSON.stringify({
        [initCurrentEditorData.tabId]: initCurrentEditorData
    }))
    const initEditorDataList = Object.entries(initEditorData).reduce((pre: IEditorItem[], [key, value]) => ([...pre, { ...value }]), [])

    const [activeKey, _setActiveKey] = useState<string>(initEditorDataList[0].tabId)
    const activeKeyRef = useRef(activeKey);
    const setActiveKey = (data: string): void => {
        activeKeyRef.current = data;
        _setActiveKey(data);
    };

    const [editorStore, _seteditorStore] = useState<IEditorStore>(initEditorData)
    const editorStoreRef = useRef(editorStore);
    const seteditorStore = (data: IEditorStore): void => {
        editorStoreRef.current = data;
        _seteditorStore(data);
    };

    const initialPanes = Object.entries(editorStore).reduce((pre: IEditorItem[], [key, value]) => ([...pre, { ...value }]), [])
    const [panes, setPanes] = useState(initialPanes);
    const newTabIndex = useRef(initEditorDataList.length);

    const [columnConfig, setColumnConfig] = useState<any[]>([])
    const [dataList, setDataList] = useState<any[]>([])

    const setEditorState = (currentState: IEditorItemParams, key?: string) => {

        const targetRes: IEditorStore = {
            ...editorStoreRef.current
        }
        let currentTaskMap = {}

        if (currentState.taskMap) {
            currentTaskMap = {
                taskMap: {
                    ...editorStoreRef.current[activeKey].taskMap,
                    ...currentState.taskMap
                }
            }
        }

        targetRes[key || activeKey] = {
            ...targetRes[key || activeKey],
            ...currentState,
            ...currentTaskMap
        }

        localStorage.setItem('dataSearch2', JSON.stringify(targetRes))
        editorStoreRef.current = targetRes
        seteditorStore(targetRes)
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

    const clearEditorTaskTimerByKey = (activeKey: string) => {
        const tagList = Object.entries(editorStoreRef.current[activeKey]).reduce((pre: IEditorItem[], [key, value]) => ([...pre, { ...value }]), [])
        tagList.forEach(tag => {
            clearInterval(tag.smartTimer)
        })
        const currentTaskList = Object.entries(editorStoreRef.current[activeKey].taskMap).reduce((pre: IEditorTaskItem[], [key, value]) => ([...pre, { ...value }]), [])
        currentTaskList.forEach(task => {
            clearInterval(task.timer)
        })
    }

    const clearTaskTimer = (activeKey: string, taskId: string) => {
        const taskMap = editorStoreRef.current[activeKey].taskMap
        const task = taskMap[taskId]
        if (task) {
            clearInterval(task.timer)
        }
    }

    // 清空定时器
    useEffect(() => {
        setEditorState({
            loading: false
        })

        return () => {
            Object.entries(editorStore).forEach((item) => {
                const [key] = item
                clearEditorTaskTimerByKey(key)
            })
        }
    }, [])

    useEffect(() => {
        // 当前tab状态是runing，触发轮询
        const currentEditorStore = editorStore[activeKey]
        const currentTaskList = Object.entries(currentEditorStore.taskMap).reduce((pre: IEditorTaskItem[], [key, value]) => ([...pre, { ...value }]), [])
        currentTaskList.forEach(task => {
            if (task.status === 'running') {
                pollGetRes(task.reqId)
            }
        })
    }, [activeKey])


    const onChange = (newActiveKey: string) => {
        Object.entries(editorStore).forEach((item) => {
            const [key] = item
            if (key !== newActiveKey) {
                clearEditorTaskTimerByKey(key)
            }
        })
        setColumnConfig([])
        setDataList([])
        setActiveKey(newActiveKey);
    };

    const add = () => {
        clearEditorTaskTimerByKey(activeKey)

        const currentIndex = ++newTabIndex.current
        if (currentIndex > 10) {
            message.warn('标签数目达到限制')
        } else {
            const newActiveKey = createId();
            const title = `新查询 ${currentIndex}`
            const newPanes = [...panes];
            const initState: IEditorItem = {
                title,
                tabId: newActiveKey,
                status: 'init',
                appGroup: '',
                biz: 'your',
                smartShow: false,
                smartContent: '',
                smartTimer: undefined,
                loading: false,
                smartCache: '',
                taskMap: {}
            }
            newPanes.push(initState);
            setPanes(newPanes);
            setActiveKey(newActiveKey);

            let res: IEditorStore = {
                ...editorStore, [newActiveKey]: initState
            }

            seteditorStore(res)
            localStorage.setItem('dataSearch2', JSON.stringify(res))
        }
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

        let res = { ...editorStore }
        delete res[targetKey]
        seteditorStore(res)
        localStorage.setItem('dataSearch2', JSON.stringify(res))
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
            const { state, result, err_msg, result_url, spark_log_url, stage } = res.data
            const task: IEditorTaskItem = {
                ...editorStoreRef.current[activeKey].taskMap[task_id],
                status: state,
                step: stage,
                log: spark_log_url,
                downloadUrl: result_url,
                result,
                message: err_msg
            }
            if (state === 'success' || state === 'failure') {
                const starTime = new Date(task.startTime || '').valueOf()
                const nowTime = new Date().valueOf()
                const duration = data2Time((nowTime - starTime) / 1000)
                task.duration = duration

                setEditorState({
                    status: 'success',
                    taskMap: {
                        [task_id]: task
                    }
                })
                clearTaskTimer(activeKey, task_id)
            } else {
                setEditorState({
                    status: 'success',
                    taskMap: {
                        [task_id]: task
                    }
                })
            }
        }).catch(() => {
            clearTaskTimer(activeKey, task_id)
            message.error('查询结果失败，尝试重新运行')
            setEditorState({
                status: 'failure',
                taskMap: {
                    [task_id]: {
                        ...editorStoreRef.current[activeKey].taskMap[task_id],
                        status: 'failure',
                        step: 'end',
                    }
                }
            })
        })
    }

    const pollGetRes = (task_id: string) => {
        clearTaskTimer(activeKey, task_id)

        let timer = setInterval(() => {
            fetchData(task_id)
        }, 10000)

        setEditorState({
            taskMap: {
                [task_id]: {
                    reqId: task_id,
                    status: 'init',
                    content: editorStore[activeKey].content,
                    name: `任务${task_id}`,
                    step: 'start',
                    startTime: moment().format('YYYY-MM-DD HH:mm:ss'),
                    database: editorStore[activeKey].database,
                    table: editorStore[activeKey].table,
                    timer,
                    message: ''
                }
            }
        })
        fetchData(task_id)
    }

    const runTask = () => {
        if (editorStore[activeKey].appGroup) {
            setEditorState({ status: 'running' })
            // 运行子任务
            actionRun({
                tdw_app_group: editorStore[activeKey].appGroup,
                sql: editorStore[activeKey]?.content || '',
                dbs: editorStore[activeKey]?.database || '',
                tables: editorStore[activeKey]?.table || '',
                biz: editorStore[activeKey]?.biz || '',
            }).then(res => {
                const { err_msg, task_id } = res.data
                if (err_msg) {
                    setEditorState({
                        status: 'failure',
                    })
                    Modal.error({
                        title: '运行失败',
                        icon: <ExclamationCircleOutlined />,
                        width: 1000,
                        content: err_msg,
                        okText: '关闭',
                        // maskClosable: true
                    });
                } else if (task_id) {
                    pollGetRes(task_id)
                }
            }).catch(err => {
                setEditorState({ status: 'failure' })
            })
        } else {
            message.warning('请先选择应用组')
        }
    }

    return (
        <div className="datasearch-container fade-in d-f">
            <div className="flex1 ptb16 pl16">
                <Tabs type="editable-card" onChange={onChange} activeKey={activeKey} onEdit={onEdit}>
                    {panes.map((pane, index) => (
                        <Tabs.TabPane tab={`新查询 ${index + 1}`} key={pane.tabId} closable={index !== 0}>
                            <div className="d-f fd-c h100">
                                <div className="flex2 s0 ov-a">
                                    {
                                        editorStore[activeKey]?.loading ? <div className="codeedit-mark">
                                            <div className="d-f jc ac fd-c">
                                                <LoadingStar />
                                                <div>
                                                    结果生成中
                                            </div>
                                            </div>
                                        </div> : null
                                    }

                                    <CodeEdit
                                        value={editorStore[activeKey]?.content}
                                        onSelect={(value) => {

                                        }}
                                        onChange={(value) => {
                                            setEditorState({
                                                content: value === '' ? undefined : value,
                                                title: pane.title,
                                            })
                                        }} />
                                </div>

                                <div className="ov-h" id="showBox" style={{ height: 500 }}>
                                    <Draggable
                                        axis="y"
                                        onStart={() => { }}
                                        onDrag={(e: any) => {
                                            const showBoxDom = document.getElementById('showBox')
                                            if (showBoxDom) {
                                                const res = document.body.clientHeight - e.y
                                                showBoxDom.style.height = `${res}px`
                                            }
                                        }}
                                        onStop={() => { }}>
                                        <div className="ta-c" style={{ cursor: 'ns-resize' }}><MenuOutlined /></div>
                                    </Draggable>
                                    <div className="ptb8 plr16 bor-l bor-r b-side d-f ac jc-b bg-w">
                                        <div className="d-f ac">
                                            <Switch className="mr8"
                                                checked={editorStore[activeKey].smartShow}
                                                unCheckedChildren="正常模式"
                                                checkedChildren="智能模式" onChange={(checked) => {
                                                    setEditorState({ smartShow: checked })
                                                }} />
                                            {
                                                editorStore[activeKey].smartShow ? <InputSearch
                                                    value={editorStore[activeKey].smartContent}
                                                    isOpenSearchMatch
                                                    onChange={(value: any) => {
                                                        setEditorState({
                                                            smartContent: value,
                                                        })
                                                    }}
                                                    onSearch={(value) => {
                                                        setEditorState({
                                                            smartCache: sqlMap[value],
                                                            loading: true,
                                                        })

                                                        const timer = setInterval(() => {
                                                            const currentContent = editorStoreRef.current[activeKey].content || ''
                                                            if (editorStoreRef.current[activeKey].smartCache) {
                                                                let smartCache = editorStoreRef.current[activeKey].smartCache || ''
                                                                const tarStr = smartCache.substr(0, 20)
                                                                smartCache = smartCache.replace(tarStr, '')

                                                                setEditorState({
                                                                    smartCache,
                                                                    content: currentContent + tarStr
                                                                })
                                                            } else {
                                                                clearInterval(editorStoreRef.current[activeKey].smartTimer)
                                                                setEditorState({
                                                                    smartCache: '',
                                                                    smartTimer: undefined,
                                                                    loading: false,
                                                                })
                                                            }
                                                        }, 800)

                                                        setEditorState({
                                                            smartTimer: timer,
                                                        })
                                                    }}
                                                    options={[
                                                        'test',
                                                    ]} placeholder="智能查询" width={'240px'} /> : null
                                            }
                                        </div>
                                        <div className="d-f ac">
                                            {/* <span className="pl16">集群：</span>
                                            <Select
                                                value={editorStore[activeKey].biz}
                                                onChange={(value) => {
                                                    setEditorState({ biz: value })
                                                }} options={[]} placeholder="选择集群" style={{ width: 200 }} />

                                            <span className="pl16">应用组：</span>
                                            <InputSearch
                                                value={editorStore[activeKey].appGroup}
                                                isOpenSearchMatch
                                                onChange={(value) => {
                                                    setEditorState({ appGroup: value })
                                                    // setAppGroup(value)
                                                }} options={['test']} placeholder="应用组" width={'400px'} /> */}
                                            <Button className="ml16" type="primary" loading={editorStore[activeKey].status === 'running'} onClick={() => {
                                                runTask()
                                            }}>运行<RightCircleOutlined /></Button>
                                        </div>
                                    </div>
                                    <div className="flex1 bor b-side s0 bg-w p-r ov-a" style={{ height: 'calc(100% - 80px)' }}>
                                        <div className="pt8">
                                            <div className="tag-result bg-theme c-text-w mr16">
                                                结果
                                            </div>
                                        </div>
                                        <div className="plr16 pt8">
                                            <TaskList
                                                option={editorStore[activeKey].taskMap}
                                                onDelete={(id) => {
                                                    Modal.confirm({
                                                        title: '删除',
                                                        icon: <ExclamationCircleOutlined />,
                                                        content: '确定删除?',
                                                        okText: '确认删除',
                                                        cancelText: '取消',
                                                        okButtonProps: { danger: true },
                                                        onOk() {
                                                            let taskMap = editorStore[activeKey].taskMap
                                                            clearTaskTimer(activeKey, id)
                                                            delete taskMap[id]
                                                            setEditorState({
                                                                taskMap
                                                            })
                                                        },
                                                        onCancel() { },
                                                    });
                                                }}
                                                onRetry={(id) => {
                                                    pollGetRes(id)
                                                }}
                                            />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </Tabs.TabPane>
                    ))}
                </Tabs>
            </div>
        </div>
    )
}
