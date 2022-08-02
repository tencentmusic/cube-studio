
   export const form = {
        type: '', // 节点类型
        factory_name: '', // 工厂名
        status: 0, // 状态
        admin: '', // 负责人
        time: '' // 更新时间
    };

    export const formRolue = [
        {
            type: 'select',
            name: 'type',
            label: '节点类型',
            options: [
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
            ]
        },
        {
            type: 'input',
            name: 'factory_name',
            label: '节点工厂名'
        },
        {
            type: 'select',
            name: 'status',
            label: '状态',
            options: [
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
            ]
        },
        {
            type: 'input',
            name: 'admin',
            label: '负责人'
        },
        {
            type: 'rangePicker',
            name: 'time',
            label: '更新时间'
        }
    ];

    export  const nodeForm = {
        type: "", // 节点类型
        factory_name: "", // 节点名
    }
