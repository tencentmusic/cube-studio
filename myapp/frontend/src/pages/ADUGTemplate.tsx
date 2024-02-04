import React, { ReactText, useEffect, useRef, useState } from 'react';
import { Button, Col, Input, DatePicker, TablePaginationConfig, Row, message, Space, Menu, Dropdown, Modal, Spin, Form, Tag, Popover, Tooltip, Select, FormInstance, Upload, UploadProps, Drawer, notification, Pagination, Switch } from 'antd';
import { Content } from 'antd/lib/layout/layout';
import TitleHeader from '../components/TitleHeader/TitleHeader';
import TableBox from '../components/TableBox/TableBox';
import moment from "moment";
import { CopyOutlined, DownOutlined, ExclamationCircleOutlined, ExportOutlined, PlusOutlined, QuestionCircleOutlined, RollbackOutlined, UploadOutlined } from '@ant-design/icons'
import { useLocation, useNavigate } from 'react-router-dom';
import { getParam, getTableScroll } from '../util';
import ModalForm from '../components/ModalForm/ModalForm';
import cookies from 'js-cookie';
import { IADUGTemplateActionItem, IAppMenuItem } from '../api/interface/kubeflowInterface';
import { getADUGTemplateList, getADUGTemplateApiInfo, actionADUGTemplateDelete, getADUGTemplateDetail, actionADUGTemplateAdd, actionADUGTemplateUpdate, actionADUGTemplateSingle, actionADUGTemplateMuliple, actionADUGTemplateRetryInfo, actionADUGTemplateFavorite, actionADUGTemplateCancelFavorite, actionADUGTemplateChartOption } from '../api/kubeflowApi';
import { ColumnsType } from 'antd/lib/table';
import MixSearch, { IMixSearchParamItem } from '../components/MixSearch/MixSearch';
import DynamicForm, { calculateId, IDynamicFormConfigItem, IDynamicFormGroupConfigItem, ILinkageConfig } from '../components/DynamicForm/DynamicForm';
import ChartOptionTempalte from './ChartOptionTempalte';
import { useTranslation } from 'react-i18next';

interface fatchDataParams {
    pageConf: TablePaginationConfig
    params: any[]
    paramsMap: Record<string, any>
    sorter?: ISorterParam
    only_favorite?: boolean
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
    const [visableAdd, setVisableAdd] = useState((getParam('isVisableAdd') === 'true') || false)
    const [loadingUpdate, setLoadingUpdate] = useState(false)
    const [visableUpdate, setVisableUpdate] = useState(false)
    const [loadingDetail, setLoadingDetail] = useState(false)
    const [visableDetail, setVisableDetail] = useState(false)
    const [selectedRowKeys, setSelectedRowKeys] = useState<ReactText[]>([])
    const pageInfoInit: TablePaginationConfig = {
        current: 1,
        pageSize: PAGE_SIZE,
        total: 0,
        showSizeChanger: true,
        showQuickJumper: true,
        pageSizeOptions: [20, 50, 100, 500],
        showTotal: (total) => `${t('共')}${total}${t('条')}`,
    };
    const [pageInfo, setPageInfo] = useState<TablePaginationConfig>(pageInfoInit);
    const [currentColumns, setCurrentColumns] = useState<ColumnsType<any>>([])
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

    let customFormData: Record<string, string> = {}
    try {
        customFormData = JSON.parse(getParam('formData') || "{}")
    } catch (err) { }
    const [dynamicFormDataAdd, setDynamicFormDataAdd] = useState(customFormData)
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
    const [list_ui_type, setList_ui_type] = useState<'card' | 'table'>()
    const [list_ui_args, setList_ui_args] = useState<{
        card_width: string
        card_height: string
    }>()
    const [opsLink, setOpsLink] = useState<Array<{
        text: string
        url: string
    }>>([])
    const [listColumns, setListColumns] = useState<string[]>([])
    const [isAllDataList, _setIsAllDataList] = useState(true)
    const isAllDataListRef = useRef(isAllDataList);
    const setIsAllDataList = (data: boolean): void => {
        isAllDataListRef.current = data;
        _setIsAllDataList(data);
    };
    const [isShowCollect, _setIsShowCollect] = useState(false)
    const isShowCollectRef = useRef(isShowCollect);
    const setIsShowCollect = (data: boolean): void => {
        isShowCollectRef.current = data;
        _setIsShowCollect(data);
    };
    const [isEchartShow, setIsEchartShow] = useState(false)
    const [pageSize, setPageSize] = useState(PAGE_SIZE)
    const [listTitle, setListTitle] = useState<string>()

    const { t, i18n } = useTranslation();

    const [scrollY, setScrollY] = useState("")

    const fetchDataParams = {
        pageConf: pageInfoInit,
        params: [],
        paramsMap: filterParamsMap,
        sorter: undefined
    }

    useEffect(() => {

    }, [pageSize])

    useEffect(() => {
        setScrollY(getTableScroll())
    }, [])

    useEffect(() => {
        if (props && props.disable) {
            navigate('/404')
        }
    }, [])

    const createDyFormConfig = (data: Record<string, any>[], label_columns: Record<string, any>, description_columns: Record<string, any>): IDynamicFormConfigItem[] => {
        return data.map((item, index) => {
            let type = item['ui-type'] || 'input'
            if (type === 'select2') {
                type = 'select'
            }
            if (type === 'file') {
                type = 'fileUpload'
            }
            const label = item.label || label_columns[item.name]

            // 校验规则
            const rules = (item.validators || []).map((item: any) => {
                if (type === 'select') {
                    return item.type === 'DataRequired' ? { required: true, message: `${t('请选择')} ${label}` } : undefined
                }

                switch (item.type) {
                    case 'DataRequired':
                        return { required: true, message: `${t('请输入')} ${label}` }
                    case 'Regexp':
                        return { pattern: new RegExp(`${item.regex}`), message: `${t('请按正确的规则输入')}` }
                    case 'Length':
                        return { min: item.min || 0, max: item.max, message: `${t('请输入正确的长度')}` }
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
        setLoadingAdd(true)

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
                download_data,
                list_ui_type,
                list_ui_args,
                ops_link,
                enable_favorite,
                echart,
                page_size,
                list_title
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
                title: t('操作'),
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
                                        isShowCollectRef.current && isAllDataListRef.current ? <Menu.Item><div className="link" onClick={() => {
                                            Modal.confirm({
                                                title: t('收藏'),
                                                icon: <ExclamationCircleOutlined />,
                                                content: `${t('确定收藏')}?`,
                                                okText: t('确认收藏'),
                                                cancelText: t('取消'),
                                                onOk() {
                                                    return new Promise((resolve, reject) => {
                                                        actionADUGTemplateFavorite(`${route_base}favorite/${record[primary_key]}`)
                                                            .then((res) => {
                                                                resolve('');
                                                            })
                                                            .catch((err) => {
                                                                reject();
                                                            });
                                                    })
                                                        .then((res) => {
                                                            message.success(t('收藏成功'));
                                                            fetchData({
                                                                ...fetchDataParams,
                                                                pageConf: pageInfo,
                                                                params: filterValuesRef.current,
                                                                paramsMap: filters
                                                            });
                                                        })
                                                        .catch(() => {
                                                            message.error(t('收藏失败'));
                                                        });
                                                },
                                                onCancel() { },
                                            });
                                        }}>{t('收藏')}</div></Menu.Item> : null
                                    }
                                    {
                                        isShowCollectRef.current && !isAllDataListRef.current ? <Menu.Item><div className="link" onClick={() => {
                                            Modal.confirm({
                                                title: t('取消收藏'),
                                                icon: <ExclamationCircleOutlined />,
                                                content: `${t('确定取消收藏')}?`,
                                                okText: t('确认取消收藏'),
                                                cancelText: t('取消'),
                                                onOk() {
                                                    return new Promise((resolve, reject) => {
                                                        actionADUGTemplateCancelFavorite(`${route_base}favorite/${record[primary_key]}`)
                                                            .then((res) => {
                                                                resolve('');
                                                            })
                                                            .catch((err) => {
                                                                reject();
                                                            });
                                                    })
                                                        .then((res) => {
                                                            message.success(t('操作成功'));
                                                            fetchData({
                                                                ...fetchDataParams,
                                                                pageConf: pageInfo,
                                                                params: filterValuesRef.current,
                                                                paramsMap: filters
                                                            });
                                                        })
                                                        .catch(() => {
                                                            message.error(t('操作失败'));
                                                        });
                                                },
                                                onCancel() { },
                                            });
                                        }}>{t('取消收藏')}</div></Menu.Item> : null
                                    }
                                    {
                                        permissions.includes('can_show') ? <Menu.Item><div className="link" onClick={() => {
                                            setVisableDetail(true)
                                            fetchDataDetail(record[primary_key])
                                        }}>
                                            {t('详情')}
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

                                                setDynamicFormConfigUpdate(formConfigUpdate)
                                                setDynamicFormGroupConfigUpdate(formGroupConfigUpdate)

                                                fetchDataDetail(record[primary_key])
                                            }).catch(() => {
                                                message.warn(t('用户没有修改权限'))
                                            })
                                        }}>
                                            {t('修改')}
                                        </div></Menu.Item> : null
                                    }
                                    {
                                        permissions.includes('can_delete') ? <Menu.Item><div className="c-fail cp" onClick={() => {
                                            Modal.confirm({
                                                title: t('删除'),
                                                icon: <ExclamationCircleOutlined />,
                                                content: `${t('确定删除')}?`,
                                                okText: t('确认删除'),
                                                cancelText: t('取消'),
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
                                                            message.success(t('删除成功'));
                                                            fetchData({
                                                                ...fetchDataParams,
                                                                pageConf: pageInfo,
                                                                params: filterValuesRef.current,
                                                                paramsMap: filters
                                                            });
                                                        })
                                                        .catch(() => {
                                                            message.error(t('删除失败'));
                                                        });
                                                },
                                                onCancel() { },
                                            });
                                        }}>
                                            {t('删除')}
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
                                                    okText: t('确认'),
                                                    cancelText: t('取消'),
                                                    onOk() {
                                                        return new Promise((resolve, reject) => {
                                                            actionADUGTemplateSingle(`${route_base}action/${action.name}/${record[primary_key]}`)
                                                                .then((res) => {
                                                                    resolve(res);
                                                                })
                                                                .catch((err) => {
                                                                    reject(err);
                                                                });
                                                        })
                                                            .then((res: any) => {
                                                                message.success(t('操作成功'));

                                                                if (res.data.result.link) {
                                                                    window.open(res.data.result.link, 'bank')
                                                                }
                                                                fetchData({
                                                                    ...fetchDataParams,
                                                                    pageConf: pageInfo,
                                                                    params: filterValuesRef.current,
                                                                    paramsMap: filters
                                                                });
                                                            })
                                                            .catch(() => {
                                                                message.error(t('操作失败'));
                                                            });
                                                    },
                                                    onCancel() { },
                                                });
                                            }}>
                                                {t(`${action.text}`)}
                                            </div></Menu.Item>
                                        })
                                    }
                                </Menu>}>
                                    <div className="link">{t('更多')}<DownOutlined /></div>
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
            if (customFormData && Object.keys(customFormData).length) {
                const reTryInfoQuene = (Object.keys(customFormData) || []).filter(key => customFormData[key] && addColumnsMap[key] && addColumnsMap[key].retry_info)
                let reTryInfoFlag = reTryInfoQuene.length

                const handleReTryInfo = (tar: string) => {
                    reTryInfoFlag = reTryInfoFlag - 1;

                    actionADUGTemplateRetryInfo(`${route_base}_info`, { exist_add_args: tar }).then(res => {
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

                        setDynamicFormDataAdd(formReset)
                        setDynamicFormConfigAdd(formConfigAdd)
                        setDynamicFormGroupConfigAdd(formGroupConfigAdd)

                        if (reTryInfoFlag) {
                            const resTar = JSON.stringify(formReset)
                            handleReTryInfo(resTar)
                        }
                    }).catch(err => {
                        message.error(t('字段切换错误'))
                    }).finally(() => {
                        setLoadingAdd(false)
                    })
                }

                if (reTryInfoQuene.length) {
                    const formRes = customFormData
                    for (const key in formRes) {
                        if (Object.prototype.hasOwnProperty.call(formRes, key)) {
                            const value = formRes[key];
                            if (value === undefined) {
                                delete formRes[key]
                            }
                        }
                    }
                    const tar = JSON.stringify(formRes)

                    handleReTryInfo(tar)
                }
            }

            const updateColumnsMap = edit_columns.reduce((pre: any, next: any) => ({ ...pre, [next.name]: next }), {})
            edit_columns.forEach((item) => {
                if (item['ui-type'] === 'list') {
                    item.info.forEach((itemInfo: any) => {
                        updateColumnsMap[itemInfo.name] = itemInfo
                    })
                }
            })
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
                    message.error(t('filter解析异常'))
                }
            }
            const localFilter = urlFilter || localCacheFilter
            if (localFilter && localFilter.length) {
                currentFilterValues = localFilter
            }

            setListTitle(list_title)
            setPageSize(page_size)
            setIsEchartShow(echart)
            setIsShowCollect(enable_favorite)
            setOpsLink(ops_link)
            setListColumns(list_columns)
            setList_ui_type(list_ui_type)
            setList_ui_args(list_ui_args)
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
                pageConf: {
                    ...pageInfoInit,
                    pageSize: page_size
                },
                params: currentFilterValues,
                paramsMap: filters,
                sorter: undefined
            });

        }).catch(err => {
            console.log(err);
        }).finally(() => {
            setLoading(false)
            setLoadingAdd(false)
        })
    }, []);

    const formatFilterParams = (params: any[], paramsMap: Record<string, any>) => {
        let formatData = undefined
        const temlateId = getParam('id')

        formatData = {
            filters: [
                temlateId ? {
                    col: props?.model_name,
                    opr: "rel_o_m",
                    value: +temlateId
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

                    return {
                        col: param.key,
                        opr: opr,
                        value: param.value
                    }
                })
            ].filter(item => !!item),
        }
        return formatData
    }

    const fetchData = ({
        pageConf,
        params,
        paramsMap,
        sorter,
        only_favorite
    }: fatchDataParams = {
            pageConf: pageInfoInit,
            params: filterValues,
            paramsMap: filterParamsMap,
            sorter: undefined,
            only_favorite: false
        }) => {
        setLoading(true);

        const form_data = JSON.stringify({
            ...formatFilterParams(params, paramsMap),
            only_favorite,
            str_related: 1,
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
            Modal.confirm({
                title: action.confirmation,
                icon: <ExclamationCircleOutlined />,
                content: '',
                okText: t('确认'),
                cancelText: t('取消'),
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
                            message.success(t('操作成功'));
                            fetchData({
                                ...fetchDataParams,
                                pageConf: pageInfo,
                                params: filterValues,
                                sorter: sorterParam,
                                paramsMap: filterParamsMap
                            });
                        })
                        .catch(() => {
                            message.error(t('操作失败'));
                        });
                },
                onCancel() { },
            });
        } else {
            message.warn(t('请先选择'))
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
            const isXLS = file.name.indexOf('.xls') !== -1;
            const isJson = file.name.indexOf('.json') !== -1;
            const isXLSX = file.name.indexOf('.xlsx') !== -1;
            if (isCSV || isJson || isXLS || isXLSX) {
                return true
            } else {
                message.error(`文件格式支持CSV/JSON/XLS/XLSX`);
            }
        },
        onChange(info) {
            if (info.file.status === 'done') {
                notification['success']({
                    message: t('导入成功'),
                    description: JSON.stringify(info.file.response),
                });
            } else if (info.file.status === 'error') {
                notification['error']({
                    message: t('导入失败'),
                    description: JSON.stringify(info.file.response),
                });
            }
        },
    };

    return (
        <div className="fade-in h100 d-f fd-c">
            {/* 添加 */}
            <ModalForm
                title={`${t('添加')} ${labelTitle}`}
                // width={1000}
                formData={dynamicFormDataAdd}
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
                        message.success(`${t('添加')} ${labelTitle} ${t('成功')}`)
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
                        message.error(`${t('添加')} ${labelTitle} ${t('失败')}`)
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
                            message.error(t('字段切换错误'))
                        }).finally(() => {
                            setLoadingAdd(false)
                        })

                    }} formChangeRes={formChangeRes} linkageConfig={columnRelateFormat} config={dynamicFormConfigAdd} configGroup={dynamicFormGroupConfigAdd} />
                }
            </ModalForm>
            {/* 修改 */}
            <ModalForm
                title={`${t('修改')} ${labelTitle}`}
                // width={1500}
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

                    if ((updateColumnsMap[next.key] || {})['ui-type'] === 'datePicker') {
                        let value = next.value;
                        value = moment(value)
                        return { ...pre, [next.key]: value }
                    }

                    if ((updateColumnsMap[next.key] || {})['ui-type'] === 'list') {
                        const value = (next.value || []).map((item: any) => {
                            for (const listItemKey in item) {
                                if (Object.prototype.hasOwnProperty.call(item, listItemKey)) {
                                    const listItemValue = item[listItemKey];
                                    if ((updateColumnsMap[listItemKey] || {})['ui-type'] === 'datePicker') {
                                        item[listItemKey] = moment(listItemValue)
                                    }
                                }
                            }
                            return item
                        })
                        return { ...pre, [next.key]: value }
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
                            message.success(`${t('更新')} ${labelTitle} ${t('成功')}`)
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
                            message.error(`${t('更新')} ${labelTitle} ${t('失败')}`)
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
                title={`${labelTitle} ${t('详情')}`}
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
                                    <Col span={18}><pre style={{ whiteSpace: 'break-spaces' }} dangerouslySetInnerHTML={{
                                        __html: (() => {
                                            let content = item.value
                                            if (Object.prototype.toString.call(item.value) === '[object Object]' || Object.prototype.toString.call(item.value) === '[object Array]') {
                                                try {
                                                    content = JSON.stringify(item.value)
                                                } catch (error) { }
                                            }
                                            return content
                                        })()
                                    }}></pre></Col>
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
                    }}><RollbackOutlined />{t('返回')}</Button> : null
                }
                <span>{labelTitle}</span>
            </>} breadcrumbs={(props?.breadcrumbs || []).map((crumbs, idx) => {
                return <span key={`templateTitle_${props?.name}_${idx}`} className="c-icon-b fs12">/<span className="plr2">{crumbs}</span></span>
            })} >
                {
                    helpUrl ? <div className="link"><span className="pr4" onClick={() => {
                        window.open(helpUrl, 'blank')
                    }}>{t('帮助链接')}</span><QuestionCircleOutlined /></div> : null
                }
            </TitleHeader>
            <Content className="appmgmt-content bg-title h100 d-f fd-c">
                <div className="mlr16 mb16 flex1 bg-w">
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

                    {
                        isEchartShow ? <ChartOptionTempalte url={baseUrl} /> : null
                    }

                    {
                        list_ui_type !== 'card' ? <TableBox
                            cancelExportData={true}
                            tableKey={`tablebox_${location.pathname}`}
                            titleNode={<Col className="tablebox-title">
                                <div className="d-f ac">
                                    <div className="mr8">{listTitle || ''}</div>
                                    {
                                        isShowCollect ? <div className="pb2">
                                            <Switch checked={isAllDataList} checkedChildren={t('全部')} unCheckedChildren={t('我的收藏')} defaultChecked onChange={(checked) => {
                                                setIsAllDataList(checked)
                                                fetchData({
                                                    ...fetchDataParams,
                                                    pageConf: pageInfoInit,
                                                    params: filterValues,
                                                    sorter: sorterParam,
                                                    paramsMap: filterParamsMap,
                                                    only_favorite: !checked
                                                });
                                            }} />
                                        </div> : null
                                    }
                                </div>
                            </Col>}
                            buttonNode={<div className="d-f ac">
                                {
                                    opsLink && opsLink.length ? opsLink.map(config => {
                                        return <Button type="primary" className="mr16" onClick={() => {
                                            window.open(config.url, 'bank')
                                        }}>{config.text}</Button>
                                    }) : null
                                }

                                {
                                    permissions.includes('can_add') ? <Button className="mr16" type="primary" onClick={() => setVisableAdd(true)}>{t('添加')}{labelTitle}<PlusOutlined /></Button> : null
                                }
                                <div>
                                    <Dropdown overlay={<Menu>
                                        {
                                            multipleAction.map((action, index) => {
                                                return <Menu.Item key={`table_muliple_${index}`}>
                                                    <span className="link" onClick={() => handleMultiRecord(action)}>
                                                        {`${t('批量')} ${action.text}`}
                                                    </span>
                                                </Menu.Item>
                                            })
                                        }

                                    </Menu>}>
                                        <Button>{t('批量操作')} <DownOutlined /></Button>
                                    </Dropdown>
                                </div>
                                {
                                    isImportData ? <div className="d-f ml16">
                                        <Tooltip color="#fff" title={<span className="tips-content-b"><div>{t('注意：csv逗号分隔')}，</div><div>{t('第一行为列的英文名')}</div> <div className="link" onClick={() => {
                                            window.open(`${window.location.origin}${baseUrlRef.current}download_template`)
                                        }}>{(t('下载导入模板'))}</div></span>} placement="topLeft">
                                            <Upload {...uploadConfig}>
                                                <Button className="" icon={<UploadOutlined />}>{t('批量导入数据')}</Button>
                                            </Upload>
                                        </Tooltip>
                                    </div> : null
                                }
                                {
                                    isDownLoadData ? <Button className="ml16" onClick={() => {
                                        Modal.confirm({
                                            title: t('导出数据'),
                                            icon: <ExclamationCircleOutlined />,
                                            content: '',
                                            okText: t('确认导出数据'),
                                            cancelText: t('取消'),
                                            onOk() {
                                                const formatData = formatFilterParams(filterValues, filterParamsMap)
                                                const form_data = JSON.stringify(formatData)
                                                window.open(`${window.location.origin}${baseUrlRef.current}download?form_data=${form_data}`)
                                                message.success(t('导出成功'));
                                            },
                                            onCancel() { },
                                        });
                                    }}>{t('批量导出')}  <ExportOutlined /></Button> : null
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
                            }}
                            rowSelection={{
                                type: 'checkbox',
                                fixed: 'left',
                                columnWidth: 32,
                                selectedRowKeys,
                                onChange: (selectedRowKeys) => {
                                    setSelectedRowKeys(selectedRowKeys)
                                }
                            }}
                            scroll={{ x: tableWidth, y: scrollY }}
                        /> : <div className="bg-w p16">
                            <div className="d-f fw">
                                {
                                    dataList.map((row, rowIndex) => {
                                        return <div style={{ overflowY: 'auto', width: list_ui_args?.card_width, height: list_ui_args?.card_height }} key={`card${rowIndex}`} className="card-row mr16 mb16" >
                                            {
                                                Object.keys(row).map((key, itemIndex) => {
                                                    if (listColumns.includes(key)) {
                                                        return <div style={{ wordBreak: 'break-all' }} key={`row${rowIndex}${itemIndex}`} dangerouslySetInnerHTML={{ __html: row[key] }}></div>
                                                    }
                                                    return null
                                                })
                                            }
                                        </div>
                                    })
                                }
                            </div>
                            <div className="ta-r">
                                <Pagination {...pageInfo} onChange={(page, pageSize) => {
                                    fetchData({
                                        ...fetchDataParams,
                                        pageConf: {
                                            ...pageInfo,
                                            current: page,
                                            pageSize
                                        },
                                        params: filterValues,
                                        paramsMap: filterParamsMap,
                                    });
                                }} />
                            </div>
                        </div>
                    }
                </div>
            </Content>
        </div >
    );
}

