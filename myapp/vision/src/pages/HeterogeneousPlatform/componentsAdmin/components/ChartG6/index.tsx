import React, { useState, useEffect } from 'react';
import G6 from '@antv/g6';
// import insertCss from 'insert-css';
// // import { render } from 'react-dom';
// // import { Steps, Button, Menu } from 'antd';

import "./index.css"

// insertCss(`
//   #contextMenu {
//     position: absolute;
//     list-style-type: none;
//     padding: 10px 8px;
//     left: -150px;
//     background-color: rgba(255, 255, 255, 0.9);
//     border: 1px solid #e2e2e2;
//     border-radius: 4px;
//     font-size: 12px;
//     color: #545454;
//   }
//   #contextMenu li {
//     cursor: pointer;
// 		list-style-type:none;
//     list-style: none;
//     margin-left: 0px;
//   }
//   #contextMenu li:hover {
//     color: #aaa;
//   }
// `);

export default function ChartG6(props: any) {
    useEffect(() => {
        const initG6: any = () => {
               const data = props.chartsData.nodes ? props.chartsData : props.chartData
            // const data = {
            //     nodes: [
            //         {
            //             id: '1',
            //             dataType: 'alps',
            //             name: 'alps_file1',
            //         },
            //         {
            //             id: '8',
            //             dataType: 'feature_extractor',
            //             name: 'feature_extractor',
            //         },
            //     ],
            //     edges: [
            //         {
            //             source: '1',
            //             target: '2',
            //         },
            //         {
            //             source: '6',
            //             target: '8',
            //         },
            //     ],
            // };
            const contextMenu = new G6.Menu({
                getContent(evt:any) {
                    console.log(evt)
                  return `
                  <span style="color: red;width: 120px"> 删除 </span>
                 `;
                },
                handleMenuClick: (target, item: any) => {
                  if(item._cfg.model.isEdit) {
                    props.pushDelItem({...item._cfg.model})
                   }
                   graph.removeItem(item)
                },
                // offsetX and offsetY include the padding of the parent container
                // 需要加上父级容器的 padding-left 16 与自身偏移量 10
                offsetX: 16 + 10,
                // 需要加上父级容器的 padding-top 24 、画布兄弟元素高度、与自身偏移量 10
                offsetY: 0,
                // the types of items that allow the menu show up
                // 在哪些类型的元素上响应
                itemTypes: ['node'],
              });

            G6.registerNode(
                'sql',
                {
                    drawShape(cfg: any, group: any) {
                        const rect = group.addShape('rect', {
                            attrs: {
                                x: -75,
                                y: -25,
                                width: 150,
                                height: 50,
                                radius: 10,
                                stroke: '#5B8FF9',
                                fill: '#C6E5FF',
                                lineWidth: 3,
                            },
                            name: 'rect-shape',
                        });
                        if (cfg.name) {
                            group.addShape('text', {
                                attrs: {
                                    text: cfg.name,
                                    x: 0,
                                    y: 0,
                                    fill: '#00287E',
                                    fontSize: 14,
                                    textAlign: 'center',
                                    textBaseline: 'middle',
                                    fontWeight: 'bold',
                                },
                                name: 'text-shape',
                            });
                        }
                        return rect;
                    },
                    // draw: (cfg: any, group: any) => {
                    //     console.log("cfg", cfg)
                    //     return group.addShape('dom', {
                    //       attrs: {
                    //         width: 20,
                    //         height: 20,
                    //         // 传入 DOM 的 html
                    //         html: `
                    //       <div style="background-color: #fff; border: 2px solid #5B8FF9; border-radius: 5px; width: ${
                    //        "40px"
                    //       }px; height: ${"20px"}px; display: flex;">
                    //         <div style="height: 100%; width: 33%; background-color: #CDDDFD">
                    //           <img alt="img" style="line-height: 100%; padding-top: 6px; padding-left: 8px;" src="https://gw.alipayobjects.com/mdn/rms_f8c6a0/afts/img/A*Q_FQT6nwEC8AAAAAAAAAAABkARQnAQ" width="20" height="20" />  
                    //         </div>
                    //         <span style="margin:auto; padding:auto; color: #5B8FF9">${cfg.label}</span>
                    //       </div>
                    //         `,
                    //       },
                    //       draggable: true,
                    //     });
                    //   },
                },
                'single-node'
            );

            const container = document.getElementById('container');
            const width = container?.scrollWidth;
            console.log(" window.screen.availHeight",  window.screen.availHeight)
            const height = window.screen.availHeight - 60 || 800;
            const graph = new G6.Graph({
                container: 'container',
                width ,
                height,
                plugins: [contextMenu],
                layout: {
                    type: 'dagre',
                    nodesepFunc: (d: any) => {
                        if (d.id === '3') {
                            return 500;
                        }
                        return 50;
                    },
                    ranksep: 70,
                    controlPoints: true,
                },
                defaultNode: {
                    type: 'sql',
                },
                defaultEdge: {
                    type: 'cubic-vertical',
                    curveOffset: 0,
                    style: {
                        // radius: 20,
                        // offset: 45,
                        lineAppendWidth: 20,
                        endArrow: {
                            path: G6.Arrow.vee(20, 20, 25),
                            d: 35
                            },
                        lineWidth: 4,
                        stroke: '#C2C8D5',
                    },
                },
                nodeStateStyles: {
                    selected: {
                        stroke: '#d9d9d9',
                        fill: '#5394ef',
                    },
                },
                // edgeStateStyles: {
                //     active: {
                //         lineAppendWidth: 20,
                //         lineWidth: 4,
                //         stroke: '#d9d9d9',
                //         fill: '#5394ef',
                //     },
                // },
                modes: {
                    default: [
                        'drag-canvas',
                        "drag-node",
                        'zoom-canvas',
                        // "tooltip",
                        'click-select',
                        {
                            type: "tooltip",
                            formatText(model) {
                                // 提示框文本内容
                                const text = model.dataType + "";
                                return text;
                              },
                        },
                        {
                            type: "create-edge",
                            key: 'shift',
                            shouldEnd: (e:any) => {
                                // if(e.item.getEdges().length === 0) {
                                //     return true
                                // }
                               return true
                            }
                        },
                        // "tooltip"
                    ],
                },
                fitView: true,
            });


            
            graph.data(data);
            graph.render();
            props.getGraph(graph)
            props.getGraphData(data)
            graph.setMinZoom(0.2)


            // if(data.nodes && data.nodes.length === 1 && data.nodes[0].label === 'new') {
            //     console.log(graph.getEdges()[0])
            //     graph.removeItem( graph.getEdges()[0], false)
            //    setTimeout(()=> {
            //     graph.addItem('node', {
            //         id: '-1',
            //         label: "start",
            //         dataType:"start",
            //         x: 800,
            //         y: 200,
            //     }, false)

                
            //     graph.addItem('node', {
            //         id: '-2',
            //         label: "end",
            //         dataType:"end",
            //         x: 800,
            //         y: 600,
            //     }, false)
            //    }, 500)
            // } 
            
            if (typeof window !== 'undefined')
                window.onresize = () => {
                    // if (!graph || graph.get('destroyed')) return;
                    // if (!container || !container.scrollWidth || !container.scrollHeight) return;
                    // graph.changeSize(container.scrollWidth, container.scrollHeight);
                };

        }
        initG6()
    }, [props.chartData])

    return (
        <div>
            <div style={{background: "#eee"}} id="container" />
        </div>
    )
}