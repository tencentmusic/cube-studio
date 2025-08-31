import * as d3 from 'd3';
import d3TipF from 'd3-tip';
const d3Tip: any = d3TipF;

export interface ID3ToolParams {
	containerId: string;
	mainViewId: string;
	margin?: number;
}

export interface IRect {
	center: number[]
	coordList: Array<{ x: number, y: number }>
	height: number
	width: number
	x0: number
	x1: number
	x2: number
	x3: number
	y0: number
	y1: number
	y2: number
	y3: number
}

type TD3NodeDom = d3.Selection<SVGGElement, unknown, HTMLElement, any>;

export default class D3Tool {
	public container: HTMLElement;
	public scaleView: TD3NodeDom;
	public mainView: TD3NodeDom;
	public gNode: TD3NodeDom;
	public gLink: TD3NodeDom;
	public gText: TD3NodeDom;
	public gBorder: TD3NodeDom;
	public gTextCache: TD3NodeDom;
	public innerWidth: number;
	public innerHeight: number;
	public tip: any;
	public fontSize = 14;
	public zoom: d3.ZoomBehavior<any, any>;
	public currentNode: any;
	public margin: number;

	constructor({ containerId, mainViewId, margin = 0 }: ID3ToolParams) {
		const _selfThis = this;

		this.margin = margin;
		this.container = document.getElementById(containerId) as HTMLElement;
		this.mainView = d3.select(`#${mainViewId}`);
		this.scaleView = this.mainView.append('g').attr('id', 'scaleView');
		this.scaleView.attr('style', 'transition:all .3s');

		const marginObj = { top: margin, right: margin, bottom: margin, left: margin };
		const innerWidth = this.container.scrollWidth - marginObj.top - marginObj.bottom;
		const innerHeight = this.container.scrollHeight - marginObj.left - marginObj.right;
		// 设置视图区域大小
		this.mainView
			.attr('width', innerWidth)
			.attr('height', innerHeight)
			.attr('viewBox', `0 0 ${innerWidth} ${innerHeight}`)
			.attr('transform', `translate(${marginObj.left}, ${marginObj.top})`)
		// .attr('transition', 'all .3s');
		// .on('click', () => {
		// 	this.tip.hide();
		// });
		this.gTextCache = this.mainView.append('g').attr('id', 'gTextCache');
		this.gBorder = this.scaleView.append('g').attr('id', 'gBorder');
		this.gText = this.scaleView.append('g').attr('id', 'gText');
		this.gLink = this.scaleView.append('g').attr('id', 'gLink');
		this.gNode = this.scaleView
			.append('g')
			.attr('id', 'gNode')
			.attr('cursor', 'pointer')
			.attr('pointer-events', 'all')
			.attr("transform", `translate(0 ${innerHeight * 0.2}) scale(.5,.5)`);
		// 设置箭头工具
		const defs = this.scaleView.append('defs').attr('id', 'gDefs');
		defs.append('marker')
			.attr('id', 'arrowRight')
			.attr('viewBox', '-10 -10 20 20')
			.attr('refX', 10)
			.attr('refY', 0)
			.attr('markerWidth', 6)
			.attr('markerHeight', 6)
			.attr('orient', 'auto')
			.append('path')
			.attr('d', 'M -10,-10 L 10,0 L -10,10')
			.attr('fill', '#cdcdcd')
			.attr('stroke-width', 2)
			.attr('stroke', '#cdcdcd');
		// 创建提示插件
		const tip = d3Tip();
		tip.attr('class', 'd3-tip').html(function (d: any) {
			return d;
		});
		this.tip = tip;
		this.mainView.call(tip);
		// 创建缩放插件
		const zoom = d3
			.zoom()
			.scaleExtent([0.1, 2])
			// .scale(2)
			// .on("dblclick", null)
			.on('zoom', (e: any) => {
				// console.log('graphviz zoomed with event:', e);
				// fix
				// const transform = d3.event.transform;
				const transform = e.transform
				_selfThis.scaleView.attr('style', 'transition:all 0s').attr('transform', transform);
				_selfThis.tip.hide()
			}).on("end", () => {
				_selfThis.scaleView.attr('style', 'transition:all 0.3s')
			}) as d3.ZoomBehavior<any, any>;

		this.zoom = zoom;
		this.mainView
			.call(zoom as any)
			.call(zoom.transform, d3.zoomIdentity.translate(0, 0).scale(1))
			.on("dblclick.zoom", null);

		this.innerHeight = innerHeight;
		this.innerWidth = innerWidth;
	}

	/**
	 *重新调整画布尺寸
	 *
	 * @memberof CostMap
	 */
	public reSize() {
		this.mainView.attr('width', 0).attr('height', 0).attr('viewBox', `0 0 ${0} ${0}`);

		const marginObj = { top: this.margin, right: this.margin, bottom: this.margin, left: this.margin };
		const innerWidth = this.container.scrollWidth - marginObj.top - marginObj.bottom;
		const innerHeight = this.container.scrollHeight - marginObj.left - marginObj.right;

		// console.log(this.container.scrollWidth, this.container.scrollHeight);

		this.mainView
			.attr('width', innerWidth)
			.attr('height', innerHeight)
			.attr('viewBox', `0 0 ${innerWidth} ${innerHeight}`);

		this.innerHeight = innerHeight;
		this.innerWidth = innerWidth;
	}

	/**
	 *获取文字rect dom
	 *
	 * @param {string} textStr
	 * @param {number} [fontSize=14]
	 * @returns {{ width: number; height: number }}
	 * @memberof CostMap
	 */
	public getTextRect(textStr: string, fontSize = 14): { width: number; height: number } {
		const textDom = this.gTextCache.append('text').text(textStr).attr('font-size', `${fontSize}px`);
		const textBox = textDom.node()?.getBBox();
		const rect = {
			width: textBox?.width || 0,
			height: textBox?.height || 0,
		};
		textDom.remove();
		return rect;
	}

	/**
	 * 列表转树
	 * @param data 
	 */
	public list2Tree = <T>(data: Array<T>): Array<T> => {
		const res: T[] = [];
		const withoutChildrenDataList = data.map((item: any) => {
			if (item.children) {
				delete item.children;
			}
			return item;
		});
		const dataMap = this.list2Map(withoutChildrenDataList, 'ci_id');
		let root: any;
		for (let i = 0; i < withoutChildrenDataList.length; i++) {
			const item = withoutChildrenDataList[i];
			if (item.parentId !== undefined) {
				const parent = dataMap.get(item.parentId);
				if (parent.children) {
					parent.children.push(item);
				} else {
					parent.children = [item];
				}
			} else {
				root = dataMap.get(item.ci_id);
			}
		}
		res.push(root);
		return res;
	};

	/**
	 * 树形结构扁平化
	 * @param data 
	 * @param key 
	 */
	public tree2List = <T>(data: Array<T>, key = 'children'): Array<T> => {
		const quene = [...data];
		const res: T[] = [];
		while (quene.length) {
			const item: any = quene.shift();
			res.push(item);
			if (item && item[key]) {
				quene.push(...item[key]);
			}
		}
		return res;
	};

	/**
	 * 列表转字典
	 * @param list 
	 * @param key 
	 * @param isReplace 
	 */
	public list2Map = <T>(list: T[], key: string, isReplace = false): Map<string, T> => {
		const map: Map<string, T> = new Map();
		for (let i = 0; i < list.length; i++) {
			const item: any = list[i];
			if (item && item[key] && (isReplace || !map.get(item[key]))) {
				map.set(item[key], item);
			}
		}
		return map;
	};

	/**
	 * 树型结构剪枝
	 * @param data 
	 * @param key 
	 * @param id 
	 */
	public treeCutNode = <T>(data: T[], key = 'children', matchKey = 'id', id?: string): T[] => {
		if (id === undefined) return data

		const dfs = <T>(data: T[]): T[] => {
			const res: T[] = []
			for (let i = 0; i < data.length; i++) {
				const item: any = data[i];
				const tarItem = { ...item }
				if (item[key] && item[key].length) {
					if (item[matchKey] === id) {
						tarItem[key] = []
						res.push(tarItem)
					} else {
						tarItem[key] = dfs(item[key])
						res.push(tarItem)
					}
				} else {
					res.push(tarItem)
				}
			}
			return res
		}

		const treeCutAfter = dfs(data)
		return treeCutAfter
	}

	public parse2Rectangle = (coords: string = '0,0 2,0 2,2 0,2') => {
		const coordList = coords.split(' ').map(coord => {
			const [x, y] = coord.split(',')
			return { x: +x, y: +y }
		})
		const rectSourceObj: any = coordList.reduce((pre, next, currentIndex) => {
			let tar: any = { ...pre }
			tar[`x${currentIndex}`] = next.x
			tar[`y${currentIndex}`] = next.y
			return tar
		}, {})
		const width = rectSourceObj.x1 - rectSourceObj.x0
		const height = rectSourceObj.y3 - rectSourceObj.y0
		const res: IRect = {
			coordList,
			...rectSourceObj,
			width: Math.abs(width),
			height: Math.abs(height),
			center: [width / 2, height / 2]
		}
		return res
	}
}
