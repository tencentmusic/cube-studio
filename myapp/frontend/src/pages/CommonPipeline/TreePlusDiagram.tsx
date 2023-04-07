/* eslint-disable @typescript-eslint/no-invalid-this */
/* eslint-disable @typescript-eslint/no-this-alias */
import * as d3 from 'd3';
import D3Tool, { ID3ToolParams } from './D3Tool';
import { graphviz } from 'd3-graphviz';
import React from 'react';
import { INodeItem } from './TreePlusInterface';
// 防止被treeShaking
const graphvizName = graphviz.name;

interface IThemeColor {
	background: string;
	color: string;
	border: string;
	activeColor?: string;
	activeBackground?: string;
	disabled?: string
	disabledColor?: string
}

interface IPreHandleNode extends INodeItem {
	relativeKey: 'parent' | 'children'
	key: string
	level: number;
	children: IPreHandleNode[],
	parent: IPreHandleNode[]
	collectNum?: number
	collectionNodes?: IPreHandleNode[]
	expandsNumParent?: number
	collectionParent?: IPreHandleNode[]
	expandsNumChildren?: number
	collectionChildren?: IPreHandleNode[]
	isCollectionNode?: boolean
	data_fields: string
}

interface IRenderNode {
	id: string | number;
	x: string | number;
	y: string | number;
	renderId: string | number;
	data?: IPreHandleNode
	theme: IThemeColor
	renderInfo?: any
}

const NodeTypeList = ['ROOT', 'BUSINESS', 'THEME', 'COLLECT']

const ThemeColor: IThemeColor[] = [
	{ background: '#00d4001a', color: '#00d400', border: '#00d400', activeColor: '#0078d4', disabled: '#cdcdcd1a', disabledColor: '#cdcdcd' },
	{ background: '#00d4c81a', color: '#00d4c8', border: '#00d4c8', activeColor: '#0078d4', disabled: '#cdcdcd1a', disabledColor: '#cdcdcd' },
	{ background: '#0000d41a', color: '#0000d4', border: '#0000d4', activeColor: '#0078d4', disabled: '#cdcdcd1a', disabledColor: '#cdcdcd' },
	{ background: '#c800d41a', color: '#c800d4', border: '#c800d4', activeColor: '#0078d4', disabled: '#cdcdcd1a', disabledColor: '#cdcdcd' },
	{ background: '#d464001a', color: '#d46400', border: '#d46400', activeColor: '#0078d4', disabled: '#cdcdcd1a', disabledColor: '#cdcdcd' },

	{ background: '#96d4001a', color: '#96d400', border: '#96d400', activeColor: '#0078d4', disabled: '#cdcdcd1a', disabledColor: '#cdcdcd' },
	{ background: '#00d4961a', color: '#00d496', border: '#00d496', activeColor: '#0078d4', disabled: '#cdcdcd1a', disabledColor: '#cdcdcd' },
	{ background: '#0096d41a', color: '#0096d4', border: '#0096d4', activeColor: '#0078d4', disabled: '#cdcdcd1a', disabledColor: '#cdcdcd' },
	{ background: '#6400d41a', color: '#6400d4', border: '#6400d4', activeColor: '#0078d4', disabled: '#cdcdcd1a', disabledColor: '#cdcdcd' },
	{ background: '#d400001a', color: '#d40000', border: '#d40000', activeColor: '#0078d4', disabled: '#cdcdcd1a', disabledColor: '#cdcdcd' },
	{ background: '#d4c8001a', color: '#d4c800', border: '#d4c800', activeColor: '#0078d4', disabled: '#cdcdcd1a', disabledColor: '#cdcdcd' },
];

const nodeTypeThemeMap: Record<string, IThemeColor> = NodeTypeList.reduce((pre, next, index) => ({ ...pre, [next]: ThemeColor[index % ThemeColor.length] }), {})

interface IRelationDiagramProps extends ID3ToolParams {
	isCollection?: boolean
}

export default class RelationDiagram extends D3Tool {

	public dataMap = new Map<string, IPreHandleNode>();
	public dataMapByName = new Map<string, IPreHandleNode>();
	public renderNodesMap = new Map<string, IRenderNode>();
	public nodesInGraphMap = new Map<string, IPreHandleNode>();
	public nodesInCollectionMap = new Map<string, IPreHandleNode>();
	public dataNodes: IPreHandleNode[] = [];
	public activeNodeId?: string;
	public rootNode?: IPreHandleNode
	public handleNodeClick?: ((node: any) => void);
	public loadingStart?: (() => void);
	public loadingEnd?: (() => void);
	private isCollection?: boolean

	constructor({ containerId, mainViewId, margin = 0, isCollection }: IRelationDiagramProps) {
		super({ containerId, mainViewId, margin });
		this.isCollection = !!isCollection
	}

	public htmlStrEnCode = (str: string) => {
		const res = str.replace(/[\u00A0-\u9999<>\-\&\:]/g, function (i) {
			return '&#' + i.charCodeAt(0) + ';';
		})
		return res
	}

	public enCodeNodeId(id: string) {
		return `node_${id.split('').map(item => item.charCodeAt(0)).join('')}`
	}

	public deCodeNodeId(id: string) {
		const sourceId = id.replace(/^node_/, '').replaceAll('_rep_', ':').replaceAll('_mid_', '-')
		return this.htmlStrEnCode(sourceId)
	}


	public preHandleNodes(nodes: INodeItem[], relativeKey: 'children' | 'parent' = 'children'): IPreHandleNode[] {
		const nodesMapByKey = new Map<string | number, IPreHandleNode>()
		const nodesMapById = new Map<string | number, IPreHandleNode>()
		// const rootId = this.enCodeNodeId((nodes[0] || {}).nid || '')
		const rootIds = nodes.map((node, index) => `node_${relativeKey}_${index}`)
		const childrenKey = relativeKey
		const parentKey = relativeKey === 'children' ? 'parent' : 'children'

		// 处理树结构上的每一个节点
		const dfs = (nodes: INodeItem[], level = 0, upItem?: IPreHandleNode, idPath: string[] = []): IPreHandleNode[] => {
			const res: IPreHandleNode[] = [];

			for (let i = 0; i < nodes.length; i++) {
				const node = nodes[i];
				idPath.push(`${i}`)
				const encodeKey = this.htmlStrEnCode(node.nid)
				let key = `node_${relativeKey}_${idPath.join('_')}`

				const nodeCacheById = nodesMapById.get(node.nid)
				if (nodeCacheById) {
					key = nodeCacheById.key
				}
				const nodeCache = nodesMapByKey.get(key)

				let tarNode: IPreHandleNode = {
					...node,
					key,
					level,
					relativeKey,
					// 构建双向链表结构
					[parentKey]: upItem ? [upItem] : [],
					[childrenKey]: []
				} as IPreHandleNode

				// 处理已经遍历过得情况
				if (nodeCache) {
					const flag = nodeCache[parentKey].map(node => node.key).includes(upItem?.key || '')
					// const flag = false
					if (flag) {
						tarNode = nodeCache
					} else {
						tarNode = {
							...nodeCache,
							// 构建双向链表结构
							[parentKey]: [upItem, ...nodeCache[parentKey]],
						}
					}
				}

				if (node[childrenKey] && node[childrenKey]?.length) {
					dfs(node[childrenKey] || [], level + 1, tarNode, idPath)
				}

				nodesMapByKey.set(tarNode.key, tarNode)
				nodesMapById.set(tarNode.nid, tarNode)
				res.push(tarNode);

				idPath.pop()
			}
			return res;
		};

		dfs(nodes)

		// 节点重构建
		nodesMapByKey.forEach(item => {
			const currentItemList = item[parentKey]
			for (let i = 0; i < currentItemList.length; i++) {
				const currentItem = currentItemList[i];
				const itemId = currentItem.key
				const tarItem = nodesMapByKey.get(itemId)
				if (tarItem) {
					tarItem[childrenKey].push(item)
					nodesMapByKey.set(itemId, tarItem)
				}
			}
			// 更新当前节点关系
			item[parentKey] = item[parentKey].map(node => nodesMapByKey.get(node.key)) as IPreHandleNode[]
			nodesMapByKey.set(item.key, item)
		})

		const rootNode = rootIds.map(rootId => nodesMapByKey.get(rootId) as IPreHandleNode)
		const res = rootNode ? [...rootNode] : []

		return res;
	}

	public initData<T extends INodeItem>(data: T[]) {
		// 初始化
		this.nodesInCollectionMap = new Map()
		this.dataNodes = []

		console.log('data', data);

		// 这一步之后已经构建了完整的链路关系
		const preHandlePreData: IPreHandleNode[] = this.preHandleNodes(data, 'parent');
		const preHandleNextData: IPreHandleNode[] = this.preHandleNodes(data, 'children');

		console.log('preHandleNextData', preHandleNextData);

		// if (this.isCollection) {
		// 	this.handleCollectionNodes(preHandlePreData, 'parent')
		// 	this.handleCollectionNodes(preHandleNextData, 'children')
		// }

		// 合并根节点
		// const [preRoot] = preHandlePreData
		// const [nextRoot] = preHandleNextData
		// if (preRoot && nextRoot) {
		// 	preHandlePreData[0].key = `node_0`
		// 	preHandlePreData[0].children = nextRoot.children
		// 	preHandlePreData[0].expandsNumChildren = nextRoot.expandsNumChildren
		// 	preHandlePreData[0].collectionChildren = nextRoot.collectionChildren

		// 	preHandleNextData[0].key = `node_0`
		// 	preHandleNextData[0].parent = preRoot.parent
		// 	preHandleNextData[0].expandsNumParent = preRoot.expandsNumParent
		// 	preHandleNextData[0].collectionParent = preRoot.collectionParent
		// }

		// 合并根节点
		for (let i = 0; i < preHandleNextData.length; i++) {
			preHandleNextData[i].key = `node_${i}`;
			preHandleNextData[i].parent = preHandlePreData[i].parent
		}

		const rootNode = preHandleNextData[0];
		const preHandleData = preHandleNextData
		this.rootNode = rootNode;

		const preRenderData = this.preRenderDataReady(preHandleData)

		console.log('preRenderData', preRenderData);

		this.renderNode(preRenderData).then(() => {
			// if (rootNode) {
			// 	this.anchorNode(rootNode.key);
			// }
			this.centerApp()
		})
	}

	public preRenderDataReady(nodes?: IPreHandleNode[]) {

		if (nodes?.length) {
			// 扁平化
			const preData = this.tree2List(nodes, 'parent');
			const nextData = this.tree2List(nodes, 'children');
			const targetData = [...preData, ...nextData];

			// 构建图的Map
			const targetDataMap = this.list2Map(targetData, 'key');
			const targetDataMapByName = this.list2Map(targetData, 'node_name');
			const dataNodes: IPreHandleNode[] = [];
			targetDataMap.forEach((item) => {
				dataNodes.push(item);
			});
			this.dataMap = targetDataMap;
			this.dataMapByName = targetDataMapByName;
			this.dataNodes = dataNodes;
		} else {
			this.dataNodes = []
		}
		return this.dataNodes
	}

	/**
	 * 构造渲染节点
	 * @param nodes 
	 */
	public createRenderNodes(nodes: IPreHandleNode[], isDisable?: boolean): string[] {

		const res = nodes.map((node) => {
			const isInCollection = this.nodesInCollectionMap.get(node.key)
			if (!isInCollection) {
				if (node.data_fields === 'COLLECT') {
					return `${node['key']}
					[label="聚合节点，剩余${node.collectNum}个节点(双击展开) + ",
						shape=box,
						style=dashed,
						margin=0,
						id=${node.key}
					];`;
				}

				return `${node['key']}
					[label="占位符占位符占位${node.key}",
						shape=box,
						width=8,
						height=0.8,
						margin=0,
						id=${node.key}
					];`;
			} else {
				return ''
			}
		});
		return res.filter(item => !!item)
	}

	/**
	 * 构造渲染关系（边）
	 * @param nodes 
	 */
	public cerateRenderNodesRelation(nodes: IPreHandleNode[]) {
		const res: string[] = [];
		nodes.forEach((node) => {
			const { parent, children } = node;

			(parent || []).forEach((parent: any) => {
				const isInCollection = this.nodesInCollectionMap.get(node.key) || this.nodesInCollectionMap.get(parent.key)
				if (!isInCollection) {
					const tar = `
					${(parent.key)}->${(node.key)} [id="edgePre_${(parent.key)}_edge_${(node.key)}"];`;
					if (res.indexOf(tar) === -1) {
						res.push(tar);
					}
				}
			});
			(children || []).forEach((child: any) => {
				const isInCollection = this.nodesInCollectionMap.get(node.key) || this.nodesInCollectionMap.get(child.key)
				if (!isInCollection) {
					const tar = `
					${(node.key)}->${(child.key)} [id="edgePre_${(node.key)}_edge_${(child.key)}"];`;
					if (res.indexOf(tar) === -1) {
						res.push(tar);
					}
				}
			});
		});
		return res;
	}

	/**
	 * 渲染后处理，事件绑定等等
	 */
	public backRenderHandle() {
		const _selfThis = this;
		const renderNodesMap = new Map<string, IRenderNode>();
		// 去掉多余的提示信息
		d3.selectAll('title').remove()

		d3.selectAll('.node').each((item: any) => {
			try {
				const key = item.key;
				const nodeData = this.dataMap?.get(key)
				const currentColorTheme = nodeTypeThemeMap[nodeData?.data_fields || ''] || ThemeColor[0]
				const box: any = d3.selectAll(`#${key} polygon`).datum();
				const tar = {
					renderInfo: item,
					renderId: key,
					data: nodeData,
					theme: currentColorTheme,
					id: key,
					x: box.center.x,
					y: box.center.y,
				};
				renderNodesMap.set(key, tar);

				d3.selectAll(`#${key} text[fill="#000000"]`).attr('type', `mainText`);

			} catch (error) {
				console.log(error);
			}

			// 调试位置坐标
			// d3.selectAll(`#${nodeId}`)
			// 	.append('g')
			// 	.append('text')
			// 	.text(`${box.attributes.x},${box.attributes.y}`)
			// 	.attr('fill', '#ff0000')
			// 	.attr('x', box.attributes.x)
			// 	.attr('y', box.attributes.y + 300);
		});

		this.renderNodesMap = renderNodesMap;

		this.beautifulNode()

		let tipsContent: JSX.Element;

		// d3.selectAll('.node')
		// 	.on('mouseenter', function (node: any, d: any) {
		// 		const key = node.key;
		// 		const curNode = _selfThis.dataMap?.get(Number(key));
		// 		tipsContent = (
		// 			<div>
		// 				<div className="pb12 d-f jc-b ac fs16">
		// 					<strong>详情</strong>
		// 				</div>
		// 				<div>{123}</div>
		// 			</div>
		// 		);
		// 		_selfThis.tip
		// 			.offset([0, 0])
		// 			.show(ReactDOMServer.renderToString(tipsContent), this);
		// 	})
		// 	.on('mouseleave', function (node: any, d: any) {
		// 		_selfThis.tip.hide();
		// 	});

		// d3.select('.d3-tip')
		// 	.on('mouseenter', function (node: any, d: any) {
		// 		_selfThis.tip.show(ReactDOMServer.renderToString(tipsContent));
		// 	})
		// 	.on('mouseleave', function (node: any, d: any) {
		// 		_selfThis.tip.hide();
		// 	});


		// 区分单双击事件
		let timeout: any = null;
		d3.selectAll('.node[type="collect"]')
			.on('click', function (node: any, d: any) {
				clearTimeout(timeout);

				timeout = setTimeout(function () {
					_selfThis.handleNodeClick && _selfThis.handleNodeClick(node)
				}, 200)
			})
		// .on('dblclick', function (node: any, d: any) {
		// 	clearTimeout(timeout);

		// 	_selfThis.handleCollectExpand(node.key)

		// 	const rootNode = _selfThis.dataMap?.get(_selfThis.rootNode?.key || '') as IPreHandleNode
		// 	const preRenderData = _selfThis.preRenderDataReady(rootNode)
		// 	_selfThis.renderNode(preRenderData).then(() => { })
		// })

		const d3MainView = document.getElementById('d3MainView')
		if (d3MainView) {
			d3MainView.onclick = (e: any) => {
				let isNode = false
				for (let i = 0; i < e.path.length; i++) {
					const elem = e.path[i];
					if (elem.id && ~elem.id.indexOf('node_')) {
						isNode = true
						break
					}
				}
				if (!isNode) {
					_selfThis.refresh()
				}
			}
		}
	}

	public beautifulNode() {
		const _selfThis = this
		const boxs = d3.selectAll('.node polygon').data()

		console.log('boxs', boxs);

		boxs.forEach((item: any) => {
			const box = item.bbox
			const pid = item.parent.key
			const nodeData = this.dataMap?.get(pid)

			if (nodeData) {
				d3.select(`#${pid}`).remove()
				d3.select('#mainGroup')
					.append('g')
					.attr('id', pid)
					.attr('class', 'node')
					.attr('type', 'normal')
					.on('click', function (node: any, d: any) {
						const dataNode = _selfThis.renderNodesMap?.get(this.id)
						if (dataNode) {
							_selfThis.highlightRelation(dataNode)
						}
						_selfThis.handleNodeClick && _selfThis.handleNodeClick(dataNode?.data)
					})
					.append('rect')
					.attr('id', `rect_${pid}`)
					.attr('x', box.x)
					.attr('y', box.y)
					.attr('rx', box.height / 2)
					.attr('ry', box.height / 2)
					.attr('width', box.width)
					.attr('height', box.height)
					.attr('fill', '#fff')
					.attr('stroke', '#cdcdcd')
					.attr('stroke-width', 1)
					.attr('style', 'transition:all 0.3s;')

				d3.select(`#${pid}`)
					.append('path')
					.attr('d', `M${box.x} ${box.cy} L${box.x + box.width * 2 / 3} ${box.cy}`)
					.attr('stroke', '#cdcdcd')
					.attr('stroke-dasharray', '5,5')

				d3.select(`#${pid}`)
					.append('rect')
					.attr('class', 'rectBg')
					.attr('x', box.x)
					.attr('y', box.y)
					.attr('rx', box.height / 2)
					.attr('ry', box.height / 2)
					.attr('width', box.height)
					.attr('height', box.height)
					.attr('fill', nodeData.color)

				d3.select(`#${pid}`)
					.append('rect')
					.attr('class', 'rectBg')
					.attr('id', `iconRect_${pid}`)
					.attr('x', box.x + box.height / 4)
					.attr('y', box.y)
					.attr('rx', box.height / 2)
					.attr('ry', box.height / 2)
					.attr('width', box.height / 2)
					.attr('height', box.height)
					.attr('fill', nodeData.color)
					.attr('style', 'transition:all 0.3s;')

				d3.select(`#${pid}`)
					.append('g')
					.attr('id', `icon_${pid}`)
					.html(nodeData.icon)

				d3.select(`#icon_${pid} svg`)
					.attr('fill', '#fff')
					.attr('width', box.height / 2)
					.attr('height', box.height / 2)
					.attr('x', box.x + box.height / 4)
					.attr('y', box.y + box.height * 0.23)

				d3.select(`#${pid}`)
					.append('text')
					.text(nodeData.title)
					.attr('class', 'nodeType')
					.attr('x', box.x + box.height * 1.2)
					.attr('y', box.y + box.height * 0.75 / 2)
					.attr('width', box.height)
					.attr('height', box.height)
					.attr('font-weight', 'bold')
					.attr('fill', nodeData.color)
				// .attr('text-anchor', 'start')
				// .attr('dominant-baseline', 'start')

				d3.select(`#${pid}`)
					.on("mouseover", function (d) {
						d3.select(`#rect_${pid}`).attr("stroke", nodeData.color);
						d3.select(`#iconRect_${pid}`)
							.attr("rx", 0)
							.attr("ry", 0)
							.attr('x', box.x + box.height / 2);
					})
					.on("mouseout", function (d) {
						d3.select(`#rect_${pid}`).attr("stroke", "#cdcdcd");
						d3.select(`#iconRect_${pid}`)
							.attr('rx', box.height / 2)
							.attr('ry', box.height / 2)
							.attr('x', box.x + box.height / 4)
					})

				d3.select(`#${pid}`)
					.append('text')
					.text(nodeData.name)
					.attr('class', 'nodeContent')
					.attr('x', box.x + box.height * 1.2)
					.attr('y', box.y + box.height * 0.8)
					.attr('width', box.height)
					.attr('height', box.height)


				// d3.select(`#${pid}`)
				// 	.append('g')
				// 	.attr('id', `icon_status_${pid}`)
				// 	.html(nodeData.status.icon)

				// d3.select(`#icon_status_${pid} svg`)
				// 	.attr('fill', '#000')
				// 	.attr('width', box.height / 3)
				// 	.attr('height', box.height / 3)
				// 	.attr('x', box.x + box.width - box.height * 2 / 3)
				// 	.attr('y', box.y + box.height * 0.2 / 2)

				d3.select(`#${pid}`)
					.append('g')
					.attr('id', `icon_status_text_${pid}`)
					.append('text')
					.text(nodeData.status.label)
					.attr('x', box.x + box.width - box.height * 2.8 / 3)
					.attr('y', box.y + box.height * 0.6 / 2)

			}
		})
	}

	public handleCollectExpand(nodeKey: string) {
		const currentNode = this.dataMap.get(nodeKey)
		const currentNodeInGraph = currentNode?.relativeKey
		const [tarParent] = currentNode?.parent || []
		const [tarChildren] = currentNode?.children || []
		const parentNode = tarParent && this.dataMap?.get(tarParent.key)
		const childrenNode = tarChildren && this.dataMap?.get(tarChildren.key)

		if (parentNode && currentNode && currentNodeInGraph === 'children') {
			const collection = parentNode.collectionChildren || []
			const children = parentNode.children || []
			const collectionTypeNode = parentNode.children.pop() as IPreHandleNode
			const targetNode = collection.shift()

			if (targetNode) {
				// 处理在节点collect里存在关系的情况
				this.nodesInCollectionMap.delete(targetNode.key)
				const nodesQuene = [...targetNode.children]
				while (nodesQuene.length) {
					const nodeItem = nodesQuene.shift()
					if (nodeItem) {
						this.nodesInCollectionMap.delete(nodeItem.key)
						nodesQuene.push(...(nodeItem.children || []))
					}
				}

				// 处理展开关系
				currentNode.collectNum = (currentNode.collectNum || 0) - 1
				currentNode.collectionNodes = collection
				parentNode.collectionChildren = collection
				if (collection.length) {
					parentNode.children = [...children, targetNode, collectionTypeNode]
				} else {
					parentNode.children = [...children, targetNode]
				}
			}
		}

		if (childrenNode && currentNode && currentNodeInGraph === 'parent') {
			const collection = childrenNode.collectionParent || []
			const parent = childrenNode.parent || []
			const collectionTypeNode = childrenNode.parent.pop() as IPreHandleNode
			const targetNode = collection.shift()

			if (targetNode) {
				// 处理在节点collect里存在关系的情况
				this.nodesInCollectionMap.delete(targetNode.key)
				const nodesQuene = [...targetNode.parent]
				while (nodesQuene.length) {
					const nodeItem = nodesQuene.shift()
					if (nodeItem) {
						this.nodesInCollectionMap.delete(nodeItem.key)
						nodesQuene.push(...(nodeItem.parent || []))
					}
				}

				// 处理展开关系
				currentNode.collectNum = (currentNode.collectNum || 0) - 1
				currentNode.collectionNodes = collection
				childrenNode.collectionParent = collection
				if (collection.length) {
					childrenNode.parent = [...parent, targetNode, collectionTypeNode]
				} else {
					childrenNode.parent = [...parent, targetNode]
				}
			}
		}
	}

	/**
	 * 渲染节点
	 * @param nodes
	 */
	public renderNode(nodes: IPreHandleNode[], isDisable?: boolean) {
		this.loadingStart && this.loadingStart()
		// console.log('nodes', nodes);
		// console.log('nodesInCollectionMap', this.nodesInCollectionMap);
		// console.log('nodesInGraphMap', this.nodesInGraphMap);

		const nodesRender = this.createRenderNodes(nodes, isDisable)
		const nodesRenderRelation = this.cerateRenderNodesRelation(nodes);
		const dotSrc = `digraph  {
			id=mainGroup;
			rankdir = LR;
			ranksep = 1;
			nodesep = 1;
			edge [color="#cdcdcd"];
			${nodesRender.join(' ')} ${nodesRenderRelation.join(' ')}
        }`;

		const test = (d3.select('#gNode') as any)
			.graphviz({
				zoom: false,
				zoomTranslateExtent: [0, 0],
				// width: this.innerWidth,
				// height: this.innerHeight
			})
			.dot(dotSrc);
		return new Promise((resolve, reject) => {
			setTimeout(() => {
				try {
					(d3.select('#gNode') as any)
						.graphviz({
							zoom: false,
							zoomTranslateExtent: [0, 0],
							// width: this.innerWidth,
							// height: this.innerHeight
						})
						.renderDot(dotSrc);
				} catch (error) {
					console.log(error);
				}

				// 后处理
				this.backRenderHandle()

				this.loadingEnd && this.loadingEnd()
				resolve('')
			}, 0)
		})
	}

	public highlightRelation(node: IRenderNode) {

		// 全局置灰
		d3.selectAll(`.node polygon`).attr('stroke', '#cdcdcd').attr('fill', '#ffffff');
		d3.selectAll(`.node text`).attr('fill', '#cdcdcd');
		d3.selectAll(`.node .rectBg`).attr('fill', '#cdcdcd');
		d3.selectAll(`.edge path`).attr('stroke', '#cdcdcd');
		d3.selectAll(`.edge polygon`).attr('stroke', '#cdcdcd').attr('fill', '#cdcdcd');

		const cutTreeNodeParent = this.treeCutNode([node.data], 'parent', 'key', this.rootNode?.key)
		const cutTreeNodeChildren = this.treeCutNode([node.data], 'children', 'key', this.rootNode?.key)
		const nodeParentList = this.tree2List(cutTreeNodeParent, 'parent')
		const nodeChildrenList = this.tree2List(cutTreeNodeChildren, 'children')

		console.log('nodeParentList', nodeParentList);

		for (let i = 0; i < nodeParentList.length; i++) {
			const item = nodeParentList[i];

			// 高亮节点
			if (item) {
				// d3.selectAll(`#${item.key} polygon`).attr('stroke', '#0078d4').attr('fill', '#ffffff');
				// d3.selectAll(`#${item.key} text`).attr('fill', '#0078d4');
				// d3.selectAll(`#${item.key} #rect_${item.key}`).attr('stroke', '#1e1653');
				// const currentColorTheme = nodeTypeThemeMap[item.data_fields] || ThemeColor[0]
				const currentColor = item.color
				d3.selectAll(`#${item.key} .rectBg`).attr('fill', currentColor);
				d3.selectAll(`#${item.key} .nodeType`).attr('fill', currentColor);
				d3.selectAll(`#${item.key} .nodeContent`).attr('fill', '#000');
			}

			// 高亮边
			if (item && item.parent && item.parent.length) {
				for (let i = 0; i < item.parent.length; i++) {
					const par = item.parent[i];
					const edgeId = `edgePre_${par.key}_edge_${item.key}`;
					d3.selectAll(`#${edgeId} path`).attr('stroke', '#1e1653');
					d3.selectAll(`#${edgeId} polygon`).attr('stroke', '#1e1653').attr('fill', '#1e1653');
				}
			}
		}


		for (let i = 0; i < nodeChildrenList.length; i++) {
			const item = nodeChildrenList[i];
			if (item) {
				// d3.selectAll(`#${item.key} polygon`).attr('stroke', '#1e1653').attr('fill', '#ffffff');
				// d3.selectAll(`#${item.key} text`).attr('fill', '#1e1653');
				// d3.selectAll(`#${item.key} #rect_${item.key}`).attr('stroke', '#1e1653');
				// const currentColorTheme = nodeTypeThemeMap[item.data_fields] || ThemeColor[0]
				const currentColor = item.color
				d3.selectAll(`#${item.key} .rectBg`).attr('fill', currentColor);
				d3.selectAll(`#${item.key} .nodeType`).attr('fill', currentColor);
				d3.selectAll(`#${item.key} .nodeContent`).attr('fill', '#000');
			}

			if (item && item.children && item.children.length) {
				for (let i = 0; i < item.children.length; i++) {
					const child = item.children[i];
					const edgeId = `edgePre_${item.key}_edge_${child.key}`;
					d3.selectAll(`#${edgeId} path`).attr('stroke', '#1e1653')
					d3.selectAll(`#${edgeId} polygon`).attr('stroke', '#1e1653').attr('fill', '#1e1653');
				}
			}
		}
	}

	public refresh() {
		console.log('refresh');
		// todo 图的改造

		// const nodeParentList = this.tree2List([this.rootNode], 'parent')
		// const nodeChildrenList = this.tree2List([this.rootNode], 'children')
		const nodeList = this.dataNodes

		for (let i = 0; i < nodeList.length; i++) {
			const item = nodeList[i];
			if (item?.data_fields === 'COLLECT') {
				d3.selectAll(`#${item.key} polygon`).attr('stroke', '#000000')
				d3.select(`#${item.key} text`).attr('fill', '#000000');
				continue
			}

			// 高亮节点
			if (item) {
				// const currentColorTheme = nodeTypeThemeMap[item.data_fields] || ThemeColor[0]
				const currentColor = item.color
				d3.selectAll(`#${item.key} .rectBg`).attr('fill', currentColor);
				d3.selectAll(`#${item.key} .nodeType`).attr('fill', currentColor);
				d3.selectAll(`#${item.key} .nodeContent`).attr('fill', '#000');
			}


			// 高亮边
			if (item && item.parent && item.parent.length) {
				for (let i = 0; i < item.parent.length; i++) {
					const par = item.parent[i];
					const edgeId = `edgePre_${par.key}_edge_${item.key}`;
					d3.selectAll(`#${edgeId} path`).attr('stroke', '#cdcdcd');
					d3.selectAll(`#${edgeId} polygon`).attr('stroke', '#cdcdcd').attr('fill', '#cdcdcd');
				}
			}
		}

		for (let i = 0; i < nodeList.length; i++) {
			const item = nodeList[i];
			if (item?.data_fields === 'COLLECT') {
				d3.selectAll(`#${item.key} polygon`).attr('stroke', '#000000')
				d3.select(`#${item.key} text`).attr('fill', '#000000');
				continue
			}

			if (item) {
				// const currentColorTheme = nodeTypeThemeMap[item.data_fields] || ThemeColor[0]
				const currentColor = item.color
				d3.selectAll(`#${item.key} .rectBg`).attr('fill', currentColor);
				d3.selectAll(`#${item.key} .nodeType`).attr('fill', currentColor);
				d3.selectAll(`#${item.key} .nodeContent`).attr('fill', '#000');
			}

			if (item && item.children && item.children.length) {
				for (let i = 0; i < item.children.length; i++) {
					const child = item.children[i];
					const edgeId = `edgePre_${item.key}_edge_${child.key}`;
					d3.selectAll(`#${edgeId} path`).attr('stroke', '#cdcdcd')
					d3.selectAll(`#${edgeId} polygon`).attr('stroke', '#cdcdcd').attr('fill', '#cdcdcd');
				}
			}
		}
	}

	/**
	 *将某个节点移动到画布中间
	 *
	 * @param {(string)} id
	 * @memberof CostMap
	 */
	public anchorNode(id: string) {
		this.resetNode(this.activeNodeId || '');
		const graphvizDom: any = d3.select('#mainGroup').datum();
		const relativeY = graphvizDom.translation.y;

		const renderNode: any = this.renderNodesMap?.get(id);
		if (renderNode) {
			// pt转px
			const x = -renderNode.x * (96 / 72) + this.innerWidth / 2 + 100;
			// const y = renderNode.y * (96 / 72) - relativeY;
			const y = -(relativeY - -renderNode.y) * (96 / 72) + this.innerHeight / 2;
			this.mainView.call(this.zoom.transform, d3.zoomIdentity.translate(x, y).scale(1));
		}
		this.activeNode(id);
	}

	/**
	 * 将整个应用居中展示
	 *
	 * @param {(string)} id
	 * @memberof CostMap
	 */
	public centerApp() {
		this.resetNode(this.activeNodeId || '');
		const scaleView: any = d3.select('#scaleView').node();
		const scaleViewBox = scaleView.getBBox();
		const currentCenterX = scaleViewBox.width / 2;
		const viewCenterX = this.innerWidth / 2;
		const x = viewCenterX - currentCenterX

		const currentCenterY = scaleViewBox.height / 2;
		const viewCenterY = this.innerHeight / 2;
		const y = viewCenterY - currentCenterY - 250

		this.mainView.call(this.zoom.transform, d3.zoomIdentity.translate(x, y).scale(1));
	}

	public activeNode(id: string) {
		const renderNode = this.renderNodesMap?.get(id);
		const dataNode = this.dataMap?.get(id);
		if (this.activeNodeId) {
			d3.selectAll(`#${this.activeNodeId} .nodeContent`).attr('fill', '#000');
		}
		if (renderNode && dataNode) {
			this.activeNodeId = id;
			// d3.selectAll(`#${id} .nodeContent`).attr('fill', ThemeColor[dataNode.level % 5].activeColor || '');

			// console.log(d3.selectAll(`#${renderNode.renderId} text[fill="#000000"]`));
			// d3.selectAll(`#${renderNode.renderId} text`).attr('fill', ThemeColor[dataNode.level % 5].activeColor || '');
			// d3.selectAll(`#${renderNode.renderId} path`)
			// 	// .attr('fill', ThemeColor[dataNode.level].activeColor || '')
			// 	.attr('stroke', ThemeColor[dataNode.level].activeColor || '');
			// d3.selectAll(`#${renderNode.renderId} polyline`).attr(
			// 	'stroke',
			// 	ThemeColor[dataNode.level].activeColor || '',
			// );
		}
	}

	public resetNode(id: string) {
		const renderNode = this.renderNodesMap?.get(id);
		const dataNode = this.dataMap?.get(id);
		if (renderNode && dataNode) {
			d3.selectAll(`[node_id=value${renderNode.renderId}]`).attr('fill', '#000000');
			// d3.selectAll(`#${renderNode.renderId} text`).attr('fill', ThemeColor[dataNode.level % 5].color);
			// d3.selectAll(`#${renderNode.renderId} path`)
			// 	// .attr('fill', ThemeColor[dataNode.level % 5].background)
			// 	.attr('stroke', ThemeColor[dataNode.level % 5].border);
			// d3.selectAll(`#${renderNode.renderId} polyline`).attr('stroke', ThemeColor[dataNode.level % 5].border);
		}
	}
}
