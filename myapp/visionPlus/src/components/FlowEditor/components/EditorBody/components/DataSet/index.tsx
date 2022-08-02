import React from 'react';
import { Icon, Layer } from '@fluentui/react';
import { Handle, Position, NodeProps } from 'react-flow-renderer';
import Model from '../Model';
import style from './style';

const DataSet: React.FC<NodeProps> = props => {
  return (
    <>
      <Handle type="target" position={Position.Top} style={{ top: '-7px', '--hover-color': props?.data?.info?.color?.color } as any} className={style.handleStyle} />
      <div className={props.selected ? style.nodeOnSelect : style.nodeContainer} style={{ borderColor: props.selected ? props?.data?.info?.color?.color : '', backgroundColor: props.selected ? props?.data?.info?.color?.bg : '' }}>
        <div className={style.nodeIconWrapper} style={{ backgroundColor: props?.data?.info?.color?.color }}>
          {/* <div dangerouslySetInnerHTML={{ __html: '<svg t="1645412532834" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="2039" width="200" height="200"><path d="M448.544 496H256a32 32 0 0 0 0 64l146.976-0.192-233.6 233.568a32 32 0 0 0 45.248 45.248l233.664-233.632v147.264a32 32 0 1 0 64 0v-192.512a63.84 63.84 0 0 0-63.744-63.744M838.624 201.376a31.968 31.968 0 0 0-45.248 0L576 418.752V272a32 32 0 0 0-64 0v192.544c0 35.136 28.608 63.712 63.744 63.712h192.512a32 32 0 1 0 0-64l-147.488 0.224 217.856-217.856a31.968 31.968 0 0 0 0-45.248" p-id="2040"></path></svg>' }}></div> */}
          <Icon iconName="Database" className={style.nodeIcon}></Icon>
        </div>
        <div className={style.nodeContentWrapper}>
          <div className={style.nodeTitle}>{props?.data?.label}</div>
          <div className={style.nodeTips}>{props?.data?.info['template-group']} - {props?.data?.info['template']}</div>
        </div>
      </div>
      <Handle type="source" position={Position.Bottom} className={style.handleStyle} style={{ '--hover-color': props?.data?.info?.color?.color } as any} />
      {/* 配置板 */}
      <Layer hostId="hostId_model" className="data-set-layer">
        <Model model={props}></Model>
      </Layer>
    </>
  );
};

export default DataSet;
