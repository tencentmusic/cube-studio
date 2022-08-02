import { DownOutlined, ExclamationCircleOutlined } from '@ant-design/icons';
import { Dropdown, Menu, message, Modal, Space, TablePaginationConfig } from 'antd';
import { ColumnsType } from 'antd/lib/table';
import { autorun, makeAutoObservable } from 'mobx'
import { ReactText } from 'react';
import { Navigate } from 'react-router-dom';
import { IADUGTemplateActionItem, IADUGTemplateInfo, IAppMenuItem, IColumnRelateItem } from '../../api/interface/kubeflowInterface';
import { actionADUGTemplateDelete, actionADUGTemplateSingle, getADUGTemplateApiInfo, getADUGTemplateDetail, getADUGTemplateList } from '../../api/kubeflowApi';
import { calculateId, IDynamicFormConfigItem, IDynamicFormGroupConfigItem, ILinkageConfig } from '../../components/DynamicForm/DynamicForm';
import { IMixSearchParamItem } from '../../components/MixSearch/MixSearch';
import { getParam } from '../../util';

const pageInfoInit: TablePaginationConfig = {
    current: 1,
    pageSize: 10,
    total: 0,
    showSizeChanger: true,
    showQuickJumper: true,
    showTotal: (total) => `共${total}条`,
};

class CURDMainTemplateStore {
    constructor() {
        makeAutoObservable(this)
    }
    // page
    permissions: string[] = []
    helpUrl?: string
    baseUrl: string = ''
    multipleAction: IADUGTemplateActionItem[] = []
    pageId?: string
    pathname: string = ''
    model_name?: string
    related?: IAppMenuItem[]

    // form
    // 表单级联字段
    columnRelateFormat: ILinkageConfig[] = []
    dynamicFormConfigAdd: IDynamicFormConfigItem[] = []
    dynamicFormConfigUpdate: IDynamicFormConfigItem[] = []
    dynamicFormGroupConfigAdd: IDynamicFormGroupConfigItem[] = []
    dynamicFormGroupConfigUpdate: IDynamicFormGroupConfigItem[] = []
    visableDetail = false
    loadingDetail = false
    visableUpdate = false
    loadingUpdate = false
    visableAdd = false
    loadingAdd = false
    dataDetail: Array<{ label: string, value: any, key: string }> = []

    // table
    dataList: any[] = []
    loading = true
    selectedRowKeys: ReactText[] = []
    pageInfo: TablePaginationConfig = pageInfoInit
    currentColumns: ColumnsType<any> = []
    tableWidth = 1000
    sorterParam?: {
        order_column: string
        order_direction: 'desc' | 'asc'
    }
    rowKey: string = ''

    // filter
    filterParams: IMixSearchParamItem[] = []
    filterValues: Array<{ key?: ReactText, value?: ReactText }> = []
    filterParamsMap: Record<string, any> = {}

    updateColumnsMap: Record<string, any> = {}
    labelMap: Record<string, string> = {}

    fetchData = () => {
        this.loading = true
        let form_data = undefined
        const temlateId = getParam('id')

        form_data = JSON.stringify({
            str_related: 1,
            "filters": [
                temlateId ? {
                    "col": this.model_name,
                    "opr": "rel_o_m",
                    "value": +temlateId
                } : undefined,
                ...this.filterValues.filter(param => param.value !== undefined).map((param: any) => {
                    const oprList = ['ct', 'lt', 'eq', 'rel_o_m']
                    const sourceOprList: string[] = this.filterParamsMap[param.key].filter.map((item: any) => item.operator) || []
                    let opr = ''
                    for (let i = 0; i < oprList.length; i++) {
                        const currentOpr = oprList[i];
                        if (sourceOprList.includes(currentOpr)) {
                            opr = currentOpr
                            break
                        }
                    }
                    return {
                        "col": param.key,
                        "opr": opr,
                        "value": param.value
                    }
                })
            ].filter(item => !!item),
            page: (this.pageInfo.current || 1) - 1,
            page_size: this.pageInfo.pageSize || 10,
            ...this.sorterParam
        })

        getADUGTemplateList(this.baseUrl, {
            form_data,
        })
            .then((res) => {
                const { count, data } = res.data.result
                this.dataList = data
                this.selectedRowKeys = []
                this.pageInfo = { ...pageInfoInit, ...this.pageInfo, total: count }
            })
            .catch((error) => {
                console.log(error);
            })
            .finally(() => this.loading = false);
    };

    fetchDataDetail = (id: string) => {
        this.loadingDetail = true
        this.dataDetail = []
        getADUGTemplateDetail(`${this.baseUrl}${id}`)
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
                        label: this.labelMap[key] || key,
                        value: formatValue(data[key]),
                        key
                    })
                })
                this.dataDetail = detail
            })
            .catch((err: any) => { })
            .finally(() => { this.loadingDetail = false })
    }

    setConfig = (config: Record<string, any>) => {
        Object.assign(this, config)
    }

    initConfig = (config: IADUGTemplateInfo & {
        pathname: string
        related?: IAppMenuItem[]
        model_name?: string
    }) => {
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
            pathname,
            related,
            model_name
        } = config

        const columnRelated = columnRelatedFormat(column_related)
        const actionwidth = 80 || [related, permissions.includes('can_show'), permissions.includes('can_edit'), permissions.includes('can_delete')].filter(item => !!item).length * 60
        const hasAction = related || permissions.includes('can_show') || permissions.includes('can_edit') || permissions.includes('can_delete')
        const cacheColumns = localStorage.getItem(`tablebox_${pathname}`)
        const cacheColumnsWidthMap = (JSON.parse(cacheColumns || '[]')).reduce((pre: any, next: any) => ({ ...pre, [next.dataIndex]: next.width }), {});

        const actionList = Object.entries(action || {}).reduce((pre: any, [name, value]) => ([...pre, { ...value }]), [])
        const multipleAction: IADUGTemplateActionItem[] = actionList.filter((item: any) => !!item.multiple)
        const singleAction: IADUGTemplateActionItem[] = actionList.filter((item: any) => !!item.single)

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
                width: cacheColumnsWidthMap[column] || 100
            }
        })

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
                                        this.visableDetail = true
                                        this.fetchDataDetail(record.id)
                                    }}>
                                        详情
                                    </div></Menu.Item> : null
                                }
                                {
                                    permissions.includes('can_edit') ? <Menu.Item><div className="link" onClick={() => {
                                        this.visableUpdate = true
                                        getADUGTemplateApiInfo(route_base, record.id).then(res => {
                                            const { edit_columns } = res.data
                                            const formConfigUpdate: IDynamicFormConfigItem[] = createDyFormConfig(edit_columns, label_columns, description_columns)
                                            const formGroupConfigUpdate: IDynamicFormGroupConfigItem[] = edit_fieldsets.map(group => {
                                                const currentData = group.fields.map(field => this.updateColumnsMap[field]).filter(item => !!item)
                                                return {
                                                    group: group.group,
                                                    expanded: group.expanded,
                                                    config: createDyFormConfig(currentData, label_columns, description_columns)
                                                }
                                            })
                                            this.dynamicFormConfigUpdate = formConfigUpdate
                                            this.dynamicFormGroupConfigUpdate = formGroupConfigUpdate

                                            this.fetchDataDetail(record.id)
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
                                            onOk: () => {
                                                return new Promise((resolve, reject) => {
                                                    actionADUGTemplateDelete(`${route_base}${record.id}`)
                                                        .then((res) => {
                                                            resolve('');
                                                        })
                                                        .catch((err) => {
                                                            reject();
                                                        });
                                                })
                                                    .then((res) => {
                                                        message.success('删除成功');
                                                        this.fetchData();
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
                                    related?.map((item, index) => {
                                        return <Menu.Item key={`moreAction_${index}`}>
                                            <div className="link" onClick={() => {
                                                Navigate({ to: `${pathname}/${item.name}?id=${record.id}` })
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
                                                onOk: () => {
                                                    return new Promise((resolve, reject) => {
                                                        actionADUGTemplateSingle(`${route_base}/action/${action.name}/${record.id}`)
                                                            .then((res) => {
                                                                resolve('');
                                                            })
                                                            .catch((err) => {
                                                                reject();
                                                            });
                                                    })
                                                        .then((res) => {
                                                            message.success('操作成功');
                                                            this.fetchData();
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


        this.setConfig({
            columnRelateFormat: columnRelated,
            
        })
    }
}

function columnRelatedFormat(config: Record<string, IColumnRelateItem>) {
    const res: ILinkageConfig[] = Object.entries(config || {})
        .reduce((pre: any[], [key, value]) => ([...pre, {
            dep: value.src_columns,
            effect: value.des_columns.join(''),
            effectOption: value.related.reduce((ePre: any, eNext) => ({ ...ePre, [calculateId(eNext.src_value)]: eNext.des_value.map(item => ({ label: item, value: item })) }), {})
        }]), [])
    return res
}

// 表单字段处理
function createDyFormConfig(data: Record<string, any>[], labelMap: Record<string, string>, descriptionMap: Record<string, string>): IDynamicFormConfigItem[] {
    return data.map((item, index) => {
        let type = item['ui-type'] || 'input'
        if (type === 'select2') {
            type = 'select'
        }
        const label = item.label || labelMap[item.name]

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

        const list = createDyFormConfig(item.info || [], labelMap, descriptionMap)

        const res: IDynamicFormConfigItem = {
            label,
            type,
            rules,
            list,
            name: item.name,
            disable: item.disable,
            description: item.description || descriptionMap[item.name] || undefined,
            required: item.required,
            defaultValue: item.default === '' ? undefined : item.default,
            multiple: item['ui-type'] && item['ui-type'] === 'select2',
            options: (item.values || []).map((item: any) => ({ label: item.value, value: item.id })),
        }
        return res
    })
}

export default new CURDMainTemplateStore()

