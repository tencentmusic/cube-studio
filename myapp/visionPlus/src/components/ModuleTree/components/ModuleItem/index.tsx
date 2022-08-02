import React, { useState, useEffect, DragEvent } from 'react';
import { Icon } from '@fluentui/react';
import { useAppDispatch } from '@src/models/hooks';
import { updateCallout, updateCurrent, updateInfo } from '@src/models/template';
import style from './style';

interface ModelProps {
  model: any;
}

const ModuleItem: React.FC<ModelProps> = props => {
  const dispatch = useAppDispatch();
  const [modelInfo, setModelInfo] = useState(props.model);

  // 鼠标滑动事件
  const handleMouseEvent = (e: any) => {
    if (e.type === 'mouseenter') {
      dispatch(updateCurrent(e.target));
    }
    dispatch(updateCallout(e.type !== 'mouseenter'));
    dispatch(updateInfo(modelInfo));
  };

  // 鼠标点击事件
  const handleMouseDown = () => {
    dispatch(updateCallout(true));
  };

  // 拖拽开始事件
  const onDragStart = (event: DragEvent) => {
    event.dataTransfer.setData('application/reactflow', JSON.stringify(modelInfo));
    event.dataTransfer.effectAllowed = 'move';
  };

  useEffect(() => {
    setModelInfo(props.model);
  }, [props.model]);

  return (
    <div className={style.moduleItem}>
      <div className="module-card-with-hover-wrap">
        <div className="module-card-content-wrap">
          <div
            className={style.moduleListModuleCard}
            onMouseEnter={handleMouseEvent}
            onMouseLeave={handleMouseEvent}
            onMouseDown={handleMouseDown}
            onDragStart={onDragStart}
            draggable
          >
            <div>
              <Icon
                iconName="Database"
                styles={{
                  root: {
                    marginRight: '4px',
                    lineHeight: '16px',
                  },
                }}
              />
              <header>{modelInfo.name}</header>
            </div>
            <div>
              <summary>
                <span>{modelInfo.describe}</span>
              </summary>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModuleItem;
