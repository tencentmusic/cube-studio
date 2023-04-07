import { ITreeNode } from "./TreePlusInterface";

const mockData4: any = {
	data: [{
		tabName: 'tab1',
		content: [{
			groupName: 'group1',
			groupContent: [{
				label: 'xxx',
				value: 'xxx',
				type: 'iframe|test|url|md等等'
			}]
		}, {
			groupName: 'group2',
			groupContent: [{
				label: 'xxx',
				value: 'xxx',
				type: 'iframe|test|url|md等等'
			}]
		}],
		bottomButton: [{
			label: 'xxx',
			url: 'http://xxx'
		}]
	}, {
		tabName: 'tab2',
		content: [{
			groupName: 'group1',
			groupContent: [{
				label: 'xxx',
				value: 'xxx',
				type: 'iframe|test|url|md等等'
			}]
		}],
		bottomButton: [{
			label: 'xxx',
			url: 'http://xxx'
		}]
	}]
}

const mockData3: any = {
	"nid": 0,
	"pid": -1,
	"title": "数仓",
	"name": "test",
	"icon": "xxx.svg",
	"status": {
		"label": "正在运行",
		"icon": "sss.svg"
	},
	"children": [{
		"nid": 1,
		"pid": 0,
		"title": "数仓1",
		"name": "test",
		"icon": "xxx.svg",
		"status": {
			"label": "正在运行",
			"icon": "sss.svg"
		},
		"children": [{
			"nid": 3,
			"pid": 0,
			"title": "数仓3",
			"name": "test",
			"icon": "xxx.svg",
			"status": {
				"label": "正在运行",
				"icon": "sss.svg"
			},
			"children": []
		}]
	}, {
		"nid": 2,
		"pid": 0,
		"title": "数仓2",
		"name": "test",
		"icon": "xxx.svg",
		"status": {
			"label": "正在运行",
			"icon": "sss.svg"
		},
		"children": []
	}, {
		"nid": 3,
		"pid": 0,
		"title": "数仓3",
		"name": "test",
		"icon": "xxx.svg",
		"status": {
			"label": "正在运行",
			"icon": "sss.svg"
		},
		"children": []
	}]
}

const mockData2: ITreeNode = {
	"nid": 0,
	"pid": -1,
	"data_fields": "ROOT",
	"cn_name": "数仓",
	"en_name": "dataWareHouse",
	"status": "ENABLE",
	"create_time": "-999999999-01-01T00:00:00",
	"update_time": "-999999999-01-01T00:00:00",
	"children": [
		{
			"nid": 1,
			"pid": 0,
			"data_fields": "BUSINESS",
			"cn_name": "a1",
			"en_name": "a1",
			"status": "ENABLE",
			"create_time": "2022-09-22T01:02:52",
			"update_time": "2022-09-22T11:37:03",
			"children": [
				{
					"nid": 5,
					"pid": 1,
					"data_fields": "THEME",
					"cn_name": "3141",
					"en_name": "3141",
					"status": "ENABLE",
					"create_time": "2022-09-22T10:52:38",
					"update_time": "2022-09-22T12:01:21",
					"children": []
				},
				{
					"nid": 6,
					"pid": 1,
					"data_fields": "THEME",
					"cn_name": "2314123",
					"en_name": "313112341",
					"status": "ENABLE",
					"create_time": "2022-09-22T10:52:38",
					"update_time": "2022-09-22T12:01:22",
					"children": []
				}
			]
		},
		{
			"nid": 2,
			"pid": 0,
			"data_fields": "BUSINESS",
			"cn_name": "13141231",
			"en_name": "2131",
			"status": "ENABLE",
			"create_time": "2022-09-22T10:52:38",
			"update_time": "2022-09-22T11:37:32",
			"children": []
		},
		{
			"nid": 3,
			"pid": 0,
			"data_fields": "BUSINESS",
			"cn_name": "314",
			"en_name": "31",
			"status": "ENABLE",
			"create_time": "2022-09-22T10:52:38",
			"update_time": "2022-09-22T11:37:32",
			"children": []
		},
		{
			"nid": 4,
			"pid": 0,
			"data_fields": "BUSINESS",
			"cn_name": "4123123",
			"en_name": "21312412",
			"status": "ENABLE",
			"create_time": "2022-09-22T10:52:38",
			"update_time": "2022-09-22T11:37:32",
			"children": []
		}
	]
}

export default mockData2;
