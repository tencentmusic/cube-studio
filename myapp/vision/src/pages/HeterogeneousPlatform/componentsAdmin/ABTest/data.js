
export const form = {
    abtest_name: '', // ABTest名
    bussiness_id: "", // 业务id
    channel_id: '', // 频道id
    module_id: '', // 模块id
    client_version: '', // 客户端版本
    admin: '', // 负责人
    status: '', // 状态
    scene_name: '', // 场景
    time: '' // 更新时间
};

export const formRolue = [
    {
        type: 'input',
        name: 'abtest_name',
        label: 'ABTest名'
    },
    {
        type: 'input',
        name: 'bussiness_id',
        label: '业务id'
    },
    {
        type: 'input',
        name: 'channel_id',
        label: '频道id'
    },
    {
        type: 'input',
        name: 'module_id',
        label: '模块id'
    },
    {
        type: 'input',
        name: 'client_version',
        label: '客户端版本'
    },
    {
        type: 'input',
        name: 'admin',
        label: '负责人'
    },
    // -1-失效,0-有效,1-创建成功,2-测试发布,3-正式发布,4-公有,5-私有
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
        name: 'scene_name',
        label: '场景'
    },
    {
        type: 'rangePicker',
        name: 'time',
        label: '更新时间'
    }
];

// export  const viewForm =  [
//     {
//         type: 'input',
//         name: 'factory_name',
//         label: '边工厂名'
//     },
//     {
//         type: 'input',
//         name: 'from_node_factory',
//         label: 'Form节点'
//     },
//     {
//         type: 'input',
//         name: 'to_node_factory',
//         label: 'To节点'
//     },
//     {
//         type: 'input',
//         name: 'admin',
//         label: '负责人'
//     },
//     {
//         type: 'select',
//         name: 'status',
//         label: '状态',
//         options: [
//             {
//                 value: 0,
//                 label: "全部"
//             },
//             {
//                 value: 1,
//                 label: "有效"
//             },
//             {
//                 value: -1,
//                 label: '失效'
//             },

//         ]
//     },
//     {
//         type: 'rangePicker',
//         name: 'time',
//         label: '更新时间'
//     }
// ];