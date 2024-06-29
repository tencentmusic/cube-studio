import Icon, { ExclamationCircleOutlined, MenuOutlined, ReloadOutlined, RightCircleOutlined, SaveOutlined, SlidersOutlined, StarOutlined, StopOutlined } from '@ant-design/icons';
import { Button, Drawer, message, Modal, Select, Switch, Tabs, Tooltip } from 'antd'
import React, { useEffect, useRef, useState } from 'react'
import { actionGetDataSearchRes, actionRun, getIdexFormConfig } from '../../api/dataSearchApi';
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
import ConfigFormData, { IConfigFormDataOptionItem } from './ConfigFormData';
import { useTranslation } from 'react-i18next';
const userName = cookies.get('myapp_username')

const createId = () => {
    return Math.random().toString(36).substring(2)
}

const sqlMap: Record<string, string> = {
    'test': `CREATE TABLE dbo.EmployeePhoto
    (
        EmployeeId INT NOT NULL PRIMARY KEY,
        Photo VARBINARY(MAX) FILESTREAM NULL,
        MyRowGuidColumn UNIQUEIDENTIFIER NOT NULL ROWGUIDCOL
                        UNIQUE DEFAULT NEWID()
    );
    
    GO
    
    /*
    text_of_comment
    /* nested comment */
    */
    
    -- line comment
    
    CREATE NONCLUSTERED INDEX IX_WorkOrder_ProductID
        ON Production.WorkOrder(ProductID)
        WITH (FILLFACTOR = 80,
            PAD_INDEX = ON,
            DROP_EXISTING = ON);
    GO
    
    WHILE (SELECT AVG(ListPrice) FROM Production.Product) < $300
    BEGIN
       UPDATE Production.Product
          SET ListPrice = ListPrice * 2
       SELECT MAX(ListPrice) FROM Production.Product
       IF (SELECT MAX(ListPrice) FROM Production.Product) > $500
          BREAK
       ELSE
          CONTINUE
    END
    PRINT 'Too much for the market to bear';
    
    MERGE INTO Sales.SalesReason AS [Target]
    USING (VALUES ('Recommendation','Other'), ('Review', 'Marketing'), ('Internet', 'Promotion'))
           AS [Source] ([NewName], NewReasonType)
    ON [Target].[Name] = [Source].[NewName]
    WHEN MATCHED
    THEN UPDATE SET ReasonType = [Source].NewReasonType
    WHEN NOT MATCHED BY TARGET
    THEN INSERT ([Name], ReasonType) VALUES ([NewName], NewReasonType)
    OUTPUT $action INTO @SummaryOfChanges;
    
    SELECT ProductID, OrderQty, SUM(LineTotal) AS Total
    FROM Sales.SalesOrderDetail
    WHERE UnitPrice < $5.00
    GROUP BY ProductID, OrderQty
    ORDER BY ProductID, OrderQty
    OPTION (HASH GROUP, FAST 10);    
`,
}

export default function DataSearch() {
    const { t, i18n } = useTranslation();
    const initId = createId()
    const initCurrentEditorData: IEditorItem = {
        tabId: initId,
        title: `${t('新查询')} 1`,
        status: 'init',
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

    const [configOption, _setConfigOption] = useState<IConfigFormDataOptionItem[]>([])
    const configOptionRef = useRef(configOption);
    const setConfigOption = (data: IConfigFormDataOptionItem[]): void => {
        configOptionRef.current = data;
        _setConfigOption(data);
    };

    const initialPanes = Object.entries(editorStore).reduce((pre: IEditorItem[], [key, value]) => ([...pre, { ...value }]), [])
    const [panes, setPanes] = useState(initialPanes);
    const newTabIndex = useRef(initEditorDataList.length);

    const [columnConfig, setColumnConfig] = useState<any[]>([])
    const [dataList, setDataList] = useState<any[]>([])

    const configDataComponentRefs: any = useRef(null);

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
        getIdexFormConfig().then(res => {
            const option = res.data.result
            setConfigOption(option)
        })
    }, [])

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
            message.warn(t('标签数目达到限制'))
        } else {
            const newActiveKey = createId();
            const title = `${t('新查询')} ${currentIndex}`
            const newPanes = [...panes];
            const initState: IEditorItem = {
                title,
                tabId: newActiveKey,
                status: 'init',
                smartShow: false,
                smartContent: '',
                smartTimer: undefined,
                smartCache: '',
                loading: false,
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
            message.error(t('查询结果失败，尝试重新运行'))
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
        }, 5000)

        setEditorState({
            taskMap: {
                [task_id]: {
                    reqId: task_id,
                    status: 'init',
                    content: editorStore[activeKey].content,
                    name: `${t('任务')}${task_id}`,
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
        setEditorState({ status: 'running' })
        const customParams = configOption.map(item => item.id).reduce((pre: any, next: any) => ({ ...pre, [next]: editorStore[activeKey][next] }), {})
        // 运行子任务
        actionRun({
            sql: editorStore[activeKey]?.content || '',
            ...customParams
        }).then(res => {
            const { err_msg, task_id } = res.data
            if (err_msg) {
                setEditorState({
                    status: 'failure',
                })
                Modal.error({
                    title: t('运行失败'),
                    icon: <ExclamationCircleOutlined />,
                    width: 1000,
                    content: err_msg,
                    okText: t('关闭'),
                    // maskClosable: true
                });
            } else if (task_id) {
                pollGetRes(task_id)
            }
        }).catch(err => {
            setEditorState({ status: 'failure' })
        })
    }

    return (
        <div className="datasearch-container fade-in d-f">
            <div className="flex1 ptb16 pl16">
                <Tabs type="editable-card" onChange={onChange} activeKey={activeKey} onEdit={onEdit}>
                    {panes.map((pane, index) => (
                        <Tabs.TabPane tab={`${t('新查询')} ${index + 1}`} key={pane.tabId} closable={index !== 0}>
                            <div className="d-f fd-c h100">
                                <div className="flex2 s0 ov-a">
                                    {
                                        editorStore[activeKey]?.loading ? <div className="codeedit-mark">
                                            <div className="d-f jc ac fd-c">
                                                <LoadingStar />
                                                <div>
                                                    {t('结果生成中')}
                                                </div>
                                            </div>
                                        </div> : null
                                    }

                                    <CodeEdit
                                        value={editorStore[activeKey]?.content}
                                        onChange={(value: any) => {
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
                                                unCheckedChildren={t('正常模式')}
                                                checkedChildren={t('智能模式')} onChange={(checked) => {
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
                                                    ]} placeholder={t('AI智能生成')} width={'240px'} /> : null
                                            }
                                        </div>
                                        <div className="d-f ac">
                                            <ConfigFormData
                                                ref={configDataComponentRefs}
                                                dataValue={editorStore[activeKey]}
                                                onChange={(dataValue) => {
                                                    setEditorState(dataValue)
                                                }}
                                                onConfigChange={(option) => {
                                                    setConfigOption(option)
                                                    setEditorState({
                                                        database: 'db'
                                                    })
                                                }}
                                                option={configOptionRef.current} />
                                            <Button className="ml16" type="primary" loading={editorStore[activeKey].status === 'running'} onClick={() => {
                                                configDataComponentRefs.current.onSubmit().then((res: any) => {
                                                    runTask()
                                                })
                                            }}>{t('运行')}<RightCircleOutlined /></Button>
                                        </div>
                                    </div>
                                    <div className="flex1 bor b-side s0 bg-w p-r ov-a" style={{ height: 'calc(100% - 80px)' }}>
                                        <div className="pt8">
                                            <div className="tag-result bg-theme c-text-w mr16">
                                                {t('结果')}
                                            </div>
                                        </div>
                                        <div className="plr16 pt8">
                                            <TaskList
                                                option={editorStore[activeKey].taskMap}
                                                onDelete={(id) => {
                                                    Modal.confirm({
                                                        title: t('删除'),
                                                        icon: <ExclamationCircleOutlined />,
                                                        content: `${t('确定删除')}?`,
                                                        okText: t('确认删除'),
                                                        cancelText: t('取消'),
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
