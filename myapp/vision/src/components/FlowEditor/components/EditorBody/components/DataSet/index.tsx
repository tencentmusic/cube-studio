import React from 'react';
import { Icon, Layer } from '@fluentui/react';
import { Handle, Position, NodeProps } from 'react-flow-renderer';
import Model from '../Model';
import style from './style';

const DataSet: React.FC<NodeProps> = props => {
  return (
    <>
      <Handle type="target" position={Position.Top} className={style.handleStyle} />
      {/* <div className={props.selected ? style.nodeOnSelect : style.nodeContainer}>
        <div className={style.nodeBar}></div>
        <div className={style.nodeContent}>
          <div className={style.nodeConentTitleBar}>
            <div className={style.nodeIconWrapper}>
              <Icon iconName="Database" className={style.nodeIcon}></Icon>
              <div className={style.nodeTitle}>{props?.data?.label}</div>
            </div>
          </div>
        </div>
      </div> */}
      <div className={props.selected ? style.nodeOnSelect : style.nodeContainer} style={{ borderColor: props.selected ? props?.data?.info?.color?.color : '', backgroundColor: props.selected ? props?.data?.info?.color?.bg : '' }}>
        <div className={style.nodeIconWrapper} style={{ backgroundColor: props?.data?.info?.color?.color }}>
          <Icon iconName="Database" className={style.nodeIcon}></Icon>
        </div>
        <div className={style.nodeContentWrapper}>
          <div className={style.nodeTitle}>{props?.data?.label}</div>
          {
            props?.data?.info ? <div className={style.nodeTips}>{props?.data?.info['describe']}</div> : null
          }
        </div>
      </div>
      <Handle type="source" position={Position.Bottom} className={style.handleStyle} />
      {/* 配置板 */}
      <Layer hostId="hostId_model" className="data-set-layer">
        <Model model={props}></Model>
      </Layer>
    </>
  );
};

export default DataSet;
