import React, { useState, useEffect } from 'react';
import { Icon } from '@fluentui/react';
import style from './style';

interface ModelProps {
  model: any;
  onClick: () => void;
}

const SearchItem: React.FC<ModelProps> = props => {
  const [modelInfo, setModelInfo] = useState(props.model);

  useEffect(() => {
    setModelInfo(props.model);
  }, [props.model]);

  return (
    <div className={style.moduleItem}>
      <div className="module-card-with-hover-wrap">
        <div className="module-card-content-wrap">
          <div className={style.moduleListModuleCard} onClick={props.onClick}>
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

export default SearchItem;
