
   export const form = {
    factory_name: '', // 边工厂名
    from_node_factory: "", // Form节点
    to_node_factory: "", // To节点
    admin: '', // 负责人
    status: 0, // 状态
    time: '' // 更新时间
};

export const formRolue = [
    {
        type: 'input',
        name: 'factory_name',
        label: '边工厂名'
    },
    {
        type: 'input',
        name: 'from_node_factory',
        label: 'Form节点'
    },
    {
        type: 'input',
        name: 'to_node_factory',
        label: 'To节点'
    },
    {
        type: 'input',
        name: 'admin',
        label: '负责人'
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