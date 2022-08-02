
export const form = {
    scene_name: '', // 场景名称
    address: "", // 调用地址
    admin: '', // 负责人
    time: '' // 更新时间
};

export const formRolue = [
    {
        type: 'input',
        name: 'scene_name',
        label: '场景名称'
    },
    {
        type: 'input',
        name: 'admin',
        label: '负责人'
    },
    {
        type: 'input',
        name: 'address',
        label: '调用地址'
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