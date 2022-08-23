import React, { ReactText, useEffect, useRef, useState } from 'react';
import { Button, Col, Input, DatePicker, TablePaginationConfig, Row, message, Space, Menu, Dropdown, Modal, Spin, Form, Tag, Popover, Tooltip, Select, FormInstance, Upload, UploadProps, Drawer, notification } from 'antd';
import { Content } from 'antd/lib/layout/layout';
import TitleHeader from '../components/TitleHeader/TitleHeader';
import TableBox from '../components/TableBox/TableBox';
import moment from "moment";
import { CopyOutlined, DownOutlined, ExclamationCircleOutlined, ExportOutlined, PlusOutlined, QuestionCircleOutlined, RollbackOutlined, UploadOutlined } from '@ant-design/icons'
import { CopyToClipboard } from 'react-copy-to-clipboard';
import { useLocation, useNavigate } from 'react-router-dom';
import { getParam, getTableScroll } from '../util';
import ModalForm from '../components/ModalForm/ModalForm';
import cookies from 'js-cookie';
import { IADUGTemplateActionItem, IAppMenuItem } from '../api/interface/kubeflowInterface';
import { getADUGTemplateList, getADUGTemplateApiInfo, actionADUGTemplateDelete, getADUGTemplateDetail, actionADUGTemplateAdd, actionADUGTemplateUpdate, actionADUGTemplateSingle, actionADUGTemplateMuliple, getCustomDialog, actionADUGTemplateDownData, actionADUGTemplateRetryInfo } from '../api/kubeflowApi';
import { ColumnsType } from 'antd/lib/table';
import MixSearch, { IMixSearchParamItem } from '../components/MixSearch/MixSearch';
import DynamicForm, { calculateId, IDynamicFormConfigItem, IDynamicFormGroupConfigItem, ILinkageConfig, TDynamicFormType } from '../components/DynamicForm/DynamicForm';
import { columnConfig } from './columnConfig';

interface fatchDataParams {
    pageConf: TablePaginationConfig
    params: any[]
    paramsMap: Record<string, any>
    sorter?: ISorterParam
}

interface ISorterParam {
    order_column: string
    order_direction: 'desc' | 'asc'
}

export default function TaskListManager(props?: IAppMenuItem) {
    const PAGE_SIZE = 20;
    const navigate = useNavigate();
    const location = useLocation()
    const [dataList, setDataList] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [loadingAdd, setLoadingAdd] = useState(false)
    const [visableAdd, setVisableAdd] = useState(false)
    const [loadingUpdate, setLoadingUpdate] = useState(false)
    const [visableUpdate, setVisableUpdate] = useState(false)
    const [loadingDetail, setLoadingDetail] = useState(false)
    const [visableDetail, setVisableDetail] = useState(false)
    const [selectedRowKeys, setSelectedRowKeys] = useState<ReactText[]>([])
    const [dateInfo, setDateInfo] = useState<{
        startTime: string,
        endTime: string
    }>({
        startTime: moment().subtract(1, 'd').startOf('day').format('YYYY-MM-DD HH:mm:ss'),
        endTime: moment().subtract(1, 'd').endOf('day').format('YYYY-MM-DD HH:mm:ss')
    })
    const pageInfoInit: TablePaginationConfig = {
        current: 1,
        pageSize: PAGE_SIZE,
        total: 0,
        showSizeChanger: true,
        showQuickJumper: true,
        pageSizeOptions: [20, 50, 100, 500],
        showTotal: (total) => `共${total}条`,
    };
    const [pageInfo, setPageInfo] = useState<TablePaginationConfig>(pageInfoInit);
    const [currentColumns, setCurrentColumns] = useState<ColumnsType<any>>([])
    // const customFilter: IMixSearchParamItem[] = [
    //     { name: 'test1', type: 'input' },
    //     { name: 'test2', type: 'select', option: [{ label: 'title1', value: 'value1' }, { label: 'title2', value: 'value2' }] },
    //     { name: 'test3', type: 'input' },
    //     { name: 'test4', type: 'select' },
    // ]
    const [filterParams, setFilterParams] = useState<IMixSearchParamItem[]>([])
    const [filterValues, _setFilterValues] = useState<Array<{ key: ReactText | undefined, value: ReactText | undefined }>>([])
    const filterValuesRef = useRef(filterValues);
    const setFilterValues = (data: Array<{ key: ReactText | undefined, value: ReactText | undefined }>): void => {
        filterValuesRef.current = data;
        _setFilterValues(data);
    };
    const [dynamicFormConfigAdd, setDynamicFormConfigAdd] = useState<IDynamicFormConfigItem[]>([])
    const [dynamicFormConfigUpdate, setDynamicFormConfigUpdate] = useState<IDynamicFormConfigItem[]>([])
    const [dynamicFormGroupConfigAdd, setDynamicFormGroupConfigAdd] = useState<IDynamicFormGroupConfigItem[]>([])
    const [dynamicFormGroupConfigUpdate, setDynamicFormGroupConfigUpdate] = useState<IDynamicFormGroupConfigItem[]>([])
    const [dynamicFormDataAdd, setDynamicFormDataAdd] = useState({})
    const [updateColumnsMap, setUpdateColumnsMap] = useState<Record<string, any>>({})
    const [labelMap, _setLabelMap] = useState<Record<string, string>>({})
    const labelMapRef = useRef(labelMap);
    const setLabelMap = (data: Record<string, string>): void => {
        labelMapRef.current = data;
        _setLabelMap(data);
    };
    const [dataDetail, setDataDetail] = useState<Array<{ label: string, value: any, key: string }>>([])
    const [tableWidth, setTableWidth] = useState(1000)
    const [permissions, setPermissions] = useState<string[]>([])
    // const [tips, setTips] = useState<Array<{ label: string, value: any }>>([])
    const [filterParamsMap, setFilterParamsMap] = useState<Record<string, any>>({})
    const [helpUrl, setHelpUrl] = useState<string | null>()

    const [baseUrl, _setBaseUrl] = useState<string>()
    const baseUrlRef = useRef(baseUrl);
    const setBaseUrl = (data: string): void => {
        baseUrlRef.current = data;
        _setBaseUrl(data);
    };
    const [isImportData, setIsImportData] = useState(false)
    const [isDownLoadData, setIsDownLoadData] = useState(false)
    const [columnRelateFormat, setColumnRelateFormat] = useState<ILinkageConfig[]>([])
    const [multipleAction, setMultipleAction] = useState<IADUGTemplateActionItem[]>([])
    const [sorterParam, setSorterParam] = useState<{
        order_column: string
        order_direction: 'desc' | 'asc'
    }>()
    const [primaryKey, setPrimaryKey] = useState('')
    const [labelTitle, setLabelTitle] = useState('')


    const [scrollY, setScrollY] = useState("")

    const fetchDataParams = {
        pageConf: pageInfoInit,
        params: [],
        paramsMap: filterParamsMap,
        sorter: undefined
    }

    useEffect(() => {
        setScrollY(getTableScroll())
    }, [])

    useEffect(() => {
        if (props && props.disable) {
            navigate('/404')
        }
    }, [])

    // 表单字段处理
    const createDyFormConfig = (data: Record<string, any>[], label_columns: Record<string, any>, description_columns: Record<string, any>): IDynamicFormConfigItem[] => {
        return data.map((item, index) => {
            let type = item['ui-type'] || 'input'
            if (type === 'select2') {
                type = 'select'
            }
            const label = item.label || label_columns[item.name]

            // 校验规则
            const rules = (item.validators || []).map((item: any) => {
                switch (item.type) {
                    case 'DataRequired':
                        return { required: true, message: `请输入${label}` }
                    case 'Regexp':
                        return { pattern: new RegExp(`${item.regex}`), message: `请按正确的规则输入` }
                    case 'Length':
                        return { min: item.min, max: item.max, message: `请输入正确的长度` }
                    default:
                        return undefined
                }
            }).filter((item: any) => !!item)

            const list = createDyFormConfig((item.info || []), label_columns, description_columns)

            const res: IDynamicFormConfigItem = {
                label,
                type,
                rules,
                list,
                name: item.name,
                disable: item.disable,
                description: item.description || description_columns[item.name] || undefined,
                required: item.required,
                defaultValue: item.default === '' ? undefined : item.default,
                multiple: item['ui-type'] && item['ui-type'] === 'select2',
                options: (item.values || []).map((item: any) => ({ label: item.value, value: item.id })),
                data: { ...item }
            }
            return res
        })
    }

    useEffect(() => {
        const targetId = getParam('targetId')
        const url = targetId ? `/dimension_remote_table_modelview/${targetId}/api/` : props?.url

        getADUGTemplateApiInfo(url).then(res => {
            const {
                list_columns,
                label_columns,
                filters,
                add_columns,
                edit_columns,
                permissions,
                description_columns,
                add_fieldsets,
                edit_fieldsets,
                help_url,
                order_columns,
                action,
                route_base,
                column_related,
                primary_key,
                label_title,
                cols_width,
                import_data,
                download_data
            } = res.data
            const actionwidth = 80 || [props?.related, permissions.includes('can_show'), permissions.includes('can_edit'), permissions.includes('can_delete')].filter(item => !!item).length * 60
            const hasAction = props?.related || permissions.includes('can_show') || permissions.includes('can_edit') || permissions.includes('can_delete')
            const cacheColumns = localStorage.getItem(`tablebox_${location.pathname}`)
            const cacheColumnsWidthMap = (JSON.parse(cacheColumns || '[]')).reduce((pre: any, next: any) => ({ ...pre, [next.dataIndex]: next.width }), {});

            const columnRelatedFormat: ILinkageConfig[] = Object.entries(column_related || {})
                .reduce((pre: any[], [key, value]) => ([...pre, {
                    dep: value.src_columns,
                    effect: value.des_columns.join(''),
                    effectOption: value.related.reduce((ePre: any, eNext) => ({ ...ePre, [calculateId(eNext.src_value)]: eNext.des_value.map(item => ({ label: item, value: item })) }), {})
                }]), [])

            // 表格字段处理
            const listColumns = list_columns.map(column => {
                return {
                    title: label_columns[column] || column,
                    dataIndex: column,
                    key: column,
                    sorter: order_columns.includes(column) ? (a: any, b: any) => a[column] - b[column] : undefined,
                    render: (text: any, record: any) => {
                        if (text === undefined || text === '') {
                            return '-'
                        }
                        if (cols_width[column] && cols_width[column].type?.indexOf('ellip') !== -1) {
                            return <Tooltip title={<span className="tips-content" dangerouslySetInnerHTML={{ __html: text }}></span>} placement="topLeft">
                                <div className={cols_width[column].type} dangerouslySetInnerHTML={{ __html: text }}>
                                </div>
                            </Tooltip>
                        }
                        if (Object.prototype.toString.call(text) === '[object Object]') {
                            const tarRes = Object.entries(text).reduce((pre: any, [label, value]) => [...pre, { label, value }], [])
                            if (!tarRes.length) {
                                return '-'
                            }
                            return <div style={{ overflow: 'auto', maxHeight: 100 }}>
                                {
                                    tarRes.map((item: any, index: number) => {
                                        return <div key={`table_itemvalue_${index}`}>{label_columns[item.label] || item.label}:{item.value}</div>
                                    })
                                }
                            </div>
                        }
                        return <div style={{ overflow: 'auto', maxHeight: 100 }} dangerouslySetInnerHTML={{ __html: text }}></div>
                    },
                    width: cacheColumnsWidthMap[column] || (cols_width[column] && cols_width[column].width) || 100
                }
            })

            const actionList = Object.entries(action || {}).reduce((pre: any, [name, value]) => ([...pre, { ...value }]), [])
            const multipleAction: IADUGTemplateActionItem[] = actionList.filter((item: any) => !!item.multiple)
            const singleAction: IADUGTemplateActionItem[] = actionList.filter((item: any) => !!item.single)

            const tableAction: any = {
                title: '操作',
                width: actionwidth,
                dataIndex: 'handle',
                key: 'handle',
                align: 'right',
                fixed: 'right',
                render: (text: any, record: any) => {
                    return (
                        <Space size="middle">
                            {
                                hasAction ? <Dropdown overlay={<Menu>
                                    {
                                        permissions.includes('can_show') ? <Menu.Item><div className="link" onClick={() => {
                                            setVisableDetail(true)
                                            fetchDataDetail(record[primary_key])
                                        }}>
                                            详情
                                        </div></Menu.Item> : null
                                    }
                                    {
                                        permissions.includes('can_edit') ? <Menu.Item><div className="link" onClick={() => {
                                            setVisableUpdate(true)
                                            getADUGTemplateApiInfo(route_base, record[primary_key]).then(res => {
                                                const { edit_columns, label_columns, description_columns } = res.data
                                                const formConfigUpdate: IDynamicFormConfigItem[] = createDyFormConfig(edit_columns, label_columns, description_columns)
                                                const formGroupConfigUpdate: IDynamicFormGroupConfigItem[] = edit_fieldsets.map(group => {
                                                    const currentData = group.fields.map(field => updateColumnsMap[field]).filter(item => !!item)
                                                    return {
                                                        group: group.group,
                                                        expanded: group.expanded,
                                                        config: createDyFormConfig(currentData, label_columns, description_columns)
                                                    }
                                                })

                                                console.log(formConfigUpdate);
                                                setDynamicFormConfigUpdate(formConfigUpdate)
                                                setDynamicFormGroupConfigUpdate(formGroupConfigUpdate)

                                                fetchDataDetail(record[primary_key])
                                            }).catch(() => {
                                                message.warn('用户没有修改权限')
                                            })
                                        }}>
                                            修改
                                        </div></Menu.Item> : null
                                    }
                                    {
                                        permissions.includes('can_delete') ? <Menu.Item><div className="c-fail cp" onClick={() => {
                                            Modal.confirm({
                                                title: '删除',
                                                icon: <ExclamationCircleOutlined />,
                                                content: '确定删除?',
                                                okText: '确认删除',
                                                cancelText: '取消',
                                                okButtonProps: { danger: true },
                                                onOk() {
                                                    return new Promise((resolve, reject) => {
                                                        actionADUGTemplateDelete(`${route_base}${record[primary_key]}`)
                                                            .then((res) => {
                                                                resolve('');
                                                            })
                                                            .catch((err) => {
                                                                reject();
                                                            });
                                                    })
                                                        .then((res) => {
                                                            message.success('删除成功');
                                                            console.log(filterValuesRef.current);
                                                            fetchData({
                                                                ...fetchDataParams,
                                                                pageConf: pageInfo,
                                                                params: filterValuesRef.current,
                                                                paramsMap: filters
                                                            });
                                                        })
                                                        .catch(() => {
                                                            message.error('删除失败');
                                                        });
                                                },
                                                onCancel() { },
                                            });
                                        }}>
                                            删除
                                        </div></Menu.Item> : null
                                    }
                                    {
                                        props?.related?.map((item, index) => {
                                            return <Menu.Item key={`moreAction_${index}`}>
                                                <div className="link" onClick={() => {
                                                    navigate(`${location.pathname}/${item.name}?id=${record[primary_key]}`)
                                                }}>
                                                    {item.title}
                                                </div>
                                            </Menu.Item>
                                        })
                                    }
                                    {
                                        !!singleAction.length && singleAction.map((action, index) => {
                                            return <Menu.Item key={`table_action_${index}`}><div className="link" onClick={() => {
                                                Modal.confirm({
                                                    title: action.confirmation,
                                                    icon: <ExclamationCircleOutlined />,
                                                    content: '',
                                                    okText: `确认${action.confirmation}`,
                                                    cancelText: '取消',
                                                    onOk() {
                                                        return new Promise((resolve, reject) => {
                                                            actionADUGTemplateSingle(`${route_base}action/${action.name}/${record[primary_key]}`)
                                                                .then((res) => {
                                                                    resolve('');
                                                                })
                                                                .catch((err) => {
                                                                    reject();
                                                                });
                                                        })
                                                            .then((res) => {
                                                                message.success('操作成功');
                                                                fetchData({
                                                                    ...fetchDataParams,
                                                                    pageConf: pageInfo,
                                                                    params: filterValuesRef.current,
                                                                    paramsMap: filters
                                                                });
                                                            })
                                                            .catch(() => {
                                                                message.error('操作失败');
                                                            });
                                                    },
                                                    onCancel() { },
                                                });
                                            }}>
                                                {action.text}
                                            </div></Menu.Item>
                                        })
                                    }
                                </Menu>}>
                                    <div className="link">更多<DownOutlined /></div>
                                </Dropdown> : null
                            }
                        </Space>
                    );
                },
            }
            const tarColumns: React.SetStateAction<ColumnsType<any>> = [...listColumns]
            if (hasAction) {
                tarColumns.push(tableAction)
            }

            const addColumnsMap = add_columns.reduce((pre: any, next: any) => ({ ...pre, [next.name]: next }), {})
            const updateColumnsMap = edit_columns.reduce((pre: any, next: any) => ({ ...pre, [next.name]: next }), {})
            const formConfigAdd: IDynamicFormConfigItem[] = createDyFormConfig(add_columns, label_columns, description_columns)
            const formGroupConfigAdd: IDynamicFormGroupConfigItem[] = add_fieldsets.map(group => {
                const currentData = group.fields.map(field => addColumnsMap[field]).filter(item => !!item)
                return {
                    group: group.group,
                    expanded: group.expanded,
                    config: createDyFormConfig(currentData, label_columns, description_columns)
                }
            })

            const tarFilter: IMixSearchParamItem[] = Object.entries(filters)
                .reduce((pre: any, [name, value]) => {
                    return [...pre, {
                        name,
                        type: value['ui-type'] || 'input',
                        title: label_columns[name],
                        oprList: value.filter.map(item => item.operator),
                        defalutValue: value.default === '' ? undefined : value.default,
                        option: value.values ? value.values.map(item => ({ label: item.value, value: item.id })) : undefined
                    }]
                }, [])

            let currentFilterValues = Object.entries(filters)
                .reduce((pre: any, [key, value]) => {
                    return [...pre, {
                        key,
                        value: value.default
                    }]
                }, []).filter((item: any) => item.value)

            const localCacheFilter = JSON.parse(localStorage.getItem(`filter_${location.pathname}${location.search}`) || '[]')
            let urlFilter = undefined
            if (getParam('filter')) {
                try {
                    urlFilter = JSON.parse(getParam('filter') || '[]')
                } catch (error) {
                    message.error('filter解析异常')
                }
            }
            const localFilter = urlFilter || localCacheFilter
            if (localFilter && localFilter.length) {
                currentFilterValues = localFilter
            }

            setIsDownLoadData(download_data)
            setIsImportData(import_data)
            setLabelTitle(label_title)
            setPrimaryKey(primary_key)
            setColumnRelateFormat(columnRelatedFormat)
            setMultipleAction(multipleAction)
            setBaseUrl(route_base)
            setUpdateColumnsMap(updateColumnsMap)
            setFilterParamsMap(filters)
            setCurrentColumns(tarColumns)
            setFilterParams(tarFilter)
            setDynamicFormConfigAdd(formConfigAdd)
            setDynamicFormGroupConfigAdd(formGroupConfigAdd)

            setLabelMap(label_columns)
            setPermissions(permissions)
            const currentTableWidth = cacheColumns ? tarColumns.reduce((pre: any, next: any) => pre + next.width || 100, 0) + 80 : tarColumns.length * 100 + 80 + actionwidth
            setTableWidth(currentTableWidth)
            setHelpUrl(help_url)
            setFilterValues(currentFilterValues)
            fetchData({
                pageConf: pageInfoInit,
                params: currentFilterValues,
                paramsMap: filters,
                sorter: undefined
            });
        }).catch(err => {
            console.log(err);
        }).finally(() => {
            setLoading(false)
        })
    }, []);

    const fetchData = ({
        pageConf,
        params,
        paramsMap,
        sorter
    }: fatchDataParams = {
            pageConf: pageInfoInit,
            params: filterValues,
            paramsMap: filterParamsMap,
            sorter: undefined
        }) => {
        setLoading(true);
        let form_data = undefined
        const temlateId = getParam('id')

        form_data = JSON.stringify({
            str_related: 1,
            "filters": [
                temlateId ? {
                    "col": props?.model_name,
                    "opr": "rel_o_m",
                    "value": +temlateId
                } : undefined,
                ...params.filter(param => param.value !== undefined).map((param: any) => {
                    let opr = ''
                    const oprList = ['rel_o_m', 'ct', 'eq']
                    const sourceOprList: string[] = paramsMap[param.key].filter.map((item: any) => item.operator) || []

                    for (let i = 0; i < oprList.length; i++) {
                        const currentOpr = oprList[i];
                        if (sourceOprList.includes(currentOpr)) {
                            opr = currentOpr
                            break
                        }
                    }
                    // if (!isNaN(+param.value) && paramsMap[param.key] && !(paramsMap[param.key].type === 'Related' || paramsMap[param.key].type === 'QuerySelect')) {
                    //     opr = 'eq'
                    // } else {
                    //     const oprList = ['rel_o_m', 'ct', 'eq']
                    //     const sourceOprList: string[] = paramsMap[param.key].filter.map((item: any) => item.operator) || []

                    //     for (let i = 0; i < oprList.length; i++) {
                    //         const currentOpr = oprList[i];
                    //         if (sourceOprList.includes(currentOpr)) {
                    //             opr = currentOpr
                    //             break
                    //         }
                    //     }
                    // }

                    return {
                        "col": param.key,
                        "opr": opr,
                        "value": param.value
                    }
                })
            ].filter(item => !!item),
            page: (pageConf.current || 1) - 1,
            page_size: pageConf.pageSize || 10,
            ...sorter
        })

        getADUGTemplateList(baseUrlRef.current, {
            form_data,
        })
            .then((res) => {
                const { count, data } = res.data.result
                setDataList(data);
                setSelectedRowKeys([])
                setPageInfo({ ...pageInfoInit, ...pageConf, total: count });
                setSorterParam(sorter)
            })
            .catch((error) => {
                console.log(error);
            })
            .finally(() => setLoading(false));
    };

    const fetchDataDetail = (id: string) => {
        setLoadingDetail(true)
        setDataDetail([])
        getADUGTemplateDetail(`${baseUrlRef.current}${id}`)
            .then(res => {
                const data = res.data.result
                const detail: any[] = []
                const formatValue = (data: any) => {
                    if (Object.prototype.toString.call(data) === '[object Object]') {
                        return data.last_name
                    }
                    return data
                }
                Object.keys(data).forEach(key => {
                    detail.push({
                        label: labelMapRef.current[key] || key,
                        value: formatValue(data[key]),
                        key
                    })
                })
                setDataDetail(detail)
            })
            .catch(err => { })
            .finally(() => { setLoadingDetail(false) })
    }

    const handleMultiRecord = (action: IADUGTemplateActionItem) => {
        if (selectedRowKeys.length) {
            // console.log(selectedRowKeys.map((item: any) => JSON.parse(item || '{}')[primaryKey]));
            Modal.confirm({
                title: action.confirmation,
                icon: <ExclamationCircleOutlined />,
                content: '',
                okText: `确认${action.confirmation}`,
                cancelText: '取消',
                onOk() {
                    return new Promise((resolve, reject) => {
                        actionADUGTemplateMuliple(`${baseUrlRef.current}multi_action/${action.name}`, {
                            ids: selectedRowKeys.map((item: any) => JSON.parse(item || '{}')[primaryKey])
                        })
                            .then((res) => {
                                resolve('');
                            })
                            .catch((err) => {
                                reject();
                            });
                    })
                        .then((res) => {
                            message.success('操作成功');
                            fetchData({
                                ...fetchDataParams,
                                pageConf: pageInfo,
                                params: filterValues,
                                sorter: sorterParam,
                                paramsMap: filterParamsMap
                            });
                        })
                        .catch(() => {
                            message.error('操作失败');
                        });
                },
                onCancel() { },
            });
        } else {
            message.warn('请先选择')
        }
    }

    const uploadConfig: UploadProps = {
        name: 'csv_file',
        maxCount: 1,
        action: `${baseUrl}upload/`,
        headers: {
            authorization: 'authorization-text',
        },
        beforeUpload: file => {
            const isCSV = file.name.indexOf('.csv') !== -1;
            if (!isCSV) {
                message.error(`${file.name} 并不是csv文件`);
            }
            return isCSV || Upload.LIST_IGNORE;
        },
        onChange(info) {
            // if (info.file.status !== 'uploading') {
            //     console.log(info.file, info.fileList);
            // }
            if (info.file.status === 'done') {
                // message.success(`${info.file.name}，${info.file.response.message}`);
                notification['success']({
                    message: '导入成功',
                    description: JSON.stringify(info.file.response),
                });
            } else if (info.file.status === 'error') {
                // message.error(`${info.file.name} 数据导入失败`);
                notification['error']({
                    message: '导入失败',
                    description: JSON.stringify(info.file.response),
                });
            }
        },
    };

    return (
        <div className="fade-in">
            {/* 添加 */}
            <ModalForm
                title={`添加${labelTitle}`}
                width={800}
                // formData={dynamicFormDataAdd}
                loading={loadingAdd}
                visible={visableAdd}
                onCancel={() => { setVisableAdd(false) }}
                onCreate={(values, form) => {
                    setLoadingAdd(true)
                    for (const key in values) {
                        if (Object.prototype.hasOwnProperty.call(values, key)) {
                            const value = values[key];
                            if (Array.isArray(value)) {
                                if (value[0] && Object.prototype.toString.call(value[0]) === '[object Object]') {
                                    continue
                                }
                                values[key] = values[key].join(',')
                            }
                        }
                    }
                    actionADUGTemplateAdd(baseUrlRef.current, values).then((res: any) => {
                        message.success(`添加${labelTitle}成功`)
                        form.resetFields()
                        setVisableAdd(false)
                        fetchData({
                            ...fetchDataParams,
                            pageConf: pageInfo,
                            params: filterValues,
                            sorter: sorterParam,
                            paramsMap: filterParamsMap
                        });
                    }).catch(err => {
                        message.error(`添加${labelTitle}失败`)
                    }).finally(() => {
                        setLoadingAdd(false)
                    })
                }}
            >
                {
                    (form: FormInstance, formChangeRes: any) => <DynamicForm form={form} onRetryInfoChange={(value) => {
                        setLoadingAdd(true)

                        const formRes = form.getFieldsValue()
                        for (const key in formRes) {
                            if (Object.prototype.hasOwnProperty.call(formRes, key)) {
                                const value = formRes[key];
                                if (value === undefined) {
                                    delete formRes[key]
                                }
                            }
                        }
                        const tar = JSON.stringify(formRes)
                        form.resetFields()

                        actionADUGTemplateRetryInfo(`${baseUrlRef.current}_info`, { exist_add_args: tar }).then(res => {
                            const { add_columns, label_columns, description_columns, add_fieldsets } = res.data;
                            const addColumnsMap = add_columns.reduce((pre: any, next: any) => ({ ...pre, [next.name]: next }), {})
                            const formConfigAdd: IDynamicFormConfigItem[] = createDyFormConfig(add_columns, label_columns, description_columns)
                            const formGroupConfigAdd: IDynamicFormGroupConfigItem[] = add_fieldsets.map(group => {
                                const currentData = group.fields.map(field => addColumnsMap[field]).filter(item => !!item)
                                return {
                                    group: group.group,
                                    expanded: group.expanded,
                                    config: createDyFormConfig(currentData, label_columns, description_columns)
                                }
                            })
                            const formReset = add_columns.filter((item) => item.default !== '').map(column => ({ [column.name]: column.default })).reduce((pre, next) => ({ ...pre, ...next }), {})

                            form.setFieldsValue(formReset)
                            setDynamicFormConfigAdd(formConfigAdd)
                            setDynamicFormGroupConfigAdd(formGroupConfigAdd)
                        }).catch(err => {
                            message.error('字段切换错误')
                        }).finally(() => {
                            setLoadingAdd(false)
                        })

                    }} formChangeRes={formChangeRes} linkageConfig={columnRelateFormat} config={dynamicFormConfigAdd} configGroup={dynamicFormGroupConfigAdd} />
                }
            </ModalForm>
            {/* 修改 */}
            <ModalForm
                title={`修改${labelTitle}`}
                width={800}
                formData={dataDetail.reduce((pre, next) => {
                    if ((updateColumnsMap[next.key] || {})['ui-type'] === 'select') {
                        let value = next.value
                        const options = (updateColumnsMap[next.key] || {}).values || []
                        const tarIndex = options.map((item: any) => item.value).indexOf(next.value)
                        if (~tarIndex) {
                            value = options[tarIndex].id
                        }
                        return { ...pre, [next.key]: value }
                    }
                    if ((updateColumnsMap[next.key] || {})['ui-type'] === 'select2') {
                        return { ...pre, [next.key]: (next.value || '').split(',') }
                    }
                    return { ...pre, [next.key]: next.value }
                }, {})}
                loading={loadingUpdate || loadingDetail}
                visible={visableUpdate}
                onCancel={() => { setVisableUpdate(false) }}
                onCreate={(values) => {
                    setLoadingUpdate(true)
                    setDataDetail([])
                    for (const key in values) {
                        if (Object.prototype.hasOwnProperty.call(values, key)) {
                            const value = values[key];
                            if (Array.isArray(value)) {
                                if (value[0] && Object.prototype.toString.call(value[0]) === '[object Object]') {
                                    continue
                                }
                                values[key] = values[key].join(',')
                            }
                        }
                    }
                    actionADUGTemplateUpdate(`${baseUrlRef.current}${values[primaryKey]}`, values)
                        .then(res => {
                            message.success(`更新${labelTitle}成功`)
                            setVisableUpdate(false)
                            fetchData({
                                ...fetchDataParams,
                                pageConf: pageInfo,
                                params: filterValues,
                                sorter: sorterParam,
                                paramsMap: filterParamsMap
                            });
                        })
                        .catch(err => {
                            message.error(`更新${labelTitle}失败`)
                        })
                        .finally(() => { setLoadingUpdate(false) })
                }}
            >
                {
                    (form: FormInstance) => <DynamicForm form={form} primaryKey={primaryKey} config={dynamicFormConfigUpdate} linkageConfig={columnRelateFormat} configGroup={dynamicFormGroupConfigUpdate} />
                }
            </ModalForm>
            {/* 详情 */}
            <Modal
                title={`${labelTitle}详情`}
                visible={visableDetail}
                footer={null}
                width={800}
                destroyOnClose
                onCancel={() => { setVisableDetail(false) }}>
                <Spin spinning={loadingDetail}>
                    <div className="pb32" style={{ minHeight: 300 }}>
                        {
                            dataDetail.map((item, index) => {
                                return <Row className="mb16" key={`dataDetail_${index}`}>
                                    <Col span={6}><div className="ta-r"><strong>{item.label}：</strong></div></Col>
                                    <Col span={18}><pre dangerouslySetInnerHTML={{ __html: item.value }}></pre></Col>
                                </Row>
                            })
                        }
                    </div>
                </Spin>
            </Modal>
            <TitleHeader title={<>
                {
                    (props?.isSubRoute || getParam('targetId')) ? <Button className="mr16" onClick={() => {
                        navigate('/data/metadata/metadata_dimension')
                        window.location.reload()
                    }}><RollbackOutlined />返回</Button> : null
                }
                <span>{labelTitle}</span>
            </>} breadcrumbs={(props?.breadcrumbs || []).map((crumbs, idx) => {
                return <span key={`templateTitle_${props?.name}_${idx}`} className="c-hint-b fs12">/<span className="plr2">{crumbs}</span></span>
            })} >
                {
                    helpUrl ? <div className="link"><span className="pr4" onClick={() => {
                        window.open(helpUrl, 'blank')
                    }}>帮助链接</span><QuestionCircleOutlined /></div> : null
                }
            </TitleHeader>
            <Content className="appmgmt-content bg-title">
                {/* <div>
                    <img className="m32" style={{ height: 42 }} src={require('../images/star2.svg').default} alt="" />
                </div> */}
                {
                    !!filterParams.length && <MixSearch values={filterValues} params={filterParams} onChange={(values) => {
                        localStorage.setItem(`filter_${location.pathname}${location.search}`, JSON.stringify(values))
                        setFilterValues(values)
                        fetchData({
                            ...fetchDataParams,
                            pageConf: pageInfoInit,
                            params: values,
                            sorter: sorterParam,
                            paramsMap: filterParamsMap
                        });
                    }} />
                }
                {/* {
                    tips.length ? <div className="bg-module mlr24 p16">
                        {tips.map((item, index) => {
                            return <div key={`ADUGTemplate_tips_${index}`}><span>{labelMapRef.current[item.label] || item.label}：</span><span>{item.value}</span></div>
                        })}
                    </div> : null
                } */}
                <div className="m16">
                    <TableBox
                        cancelExportData={true}
                        tableKey={`tablebox_${location.pathname}`}
                        titleNode={<Col className="tablebox-title">{labelTitle}列表</Col>}
                        buttonNode={<div className="d-f ac">
                            {
                                permissions.includes('can_add') ? <Button className="mr16" type="primary" onClick={() => setVisableAdd(true)}>添加{labelTitle}<PlusOutlined /></Button> : null
                            }
                            <div>
                                <Dropdown overlay={<Menu>
                                    {
                                        multipleAction.map((action, index) => {
                                            return <Menu.Item key={`table_muliple_${index}`}>
                                                <span className="link" onClick={() => handleMultiRecord(action)}>
                                                    {`批量${action.text}`}
                                                </span>
                                            </Menu.Item>
                                        })
                                    }

                                </Menu>}>
                                    <Button>批量操作 <DownOutlined /></Button>
                                </Dropdown>
                            </div>
                            {
                                isImportData ? <div className="d-f ml16">
                                    <Tooltip color="#fff" title={<span className="tips-content-b">注意：csv逗号分隔，第一行为列的英文名 <span className="link" onClick={() => {
                                        window.open(`${window.location.origin}${baseUrlRef.current}download_template`)
                                    }}>下载导入模板</span></span>} placement="topLeft">
                                        <Upload {...uploadConfig}>
                                            <Button className="" icon={<UploadOutlined />}>批量导入数据</Button>
                                        </Upload>
                                    </Tooltip>
                                </div> : null
                            }
                            {
                                isDownLoadData ? <Button className="ml16" onClick={() => {
                                    Modal.confirm({
                                        title: '导出数据',
                                        icon: <ExclamationCircleOutlined />,
                                        content: '',
                                        okText: '确认导出数据',
                                        cancelText: '取消',
                                        onOk() {
                                            window.open(`${window.location.origin}${baseUrlRef.current}download`)
                                            message.success('导出成功');
                                        },
                                        onCancel() { },
                                    });
                                }}>批量导出  <ExportOutlined /></Button> : null
                            }

                        </div>}
                        rowKey={(record: any) => {
                            return JSON.stringify(record)
                        }}
                        columns={currentColumns}
                        loading={loading}
                        pagination={pageInfo}
                        dataSource={dataList}
                        onChange={(pageInfo: any, filters, sorter: any) => {
                            const tarSorter = sorter.order ? {
                                order_column: sorter.columnKey,
                                order_direction: sorter.order === "ascend" ? 'asc' : 'desc'
                            } as ISorterParam : undefined

                            fetchData({
                                ...fetchDataParams,
                                pageConf: pageInfo,
                                params: filterValues,
                                paramsMap: filterParamsMap,
                                sorter: tarSorter
                            });
                            // setPageInfo(pageInfo)
                        }}
                        rowSelection={{
                            type: 'checkbox',
                            fixed: 'left',
                            columnWidth: 80,
                            selectedRowKeys,
                            onChange: (selectedRowKeys) => {
                                setSelectedRowKeys(selectedRowKeys)
                            }
                        }}
                        scroll={{ x: tableWidth, y: scrollY }}
                    />
                </div>
            </Content>
        </div >
    );
}

