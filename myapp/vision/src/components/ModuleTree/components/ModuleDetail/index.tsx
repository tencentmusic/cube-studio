import React from 'react';
import { Callout, Stack, Label, PrimaryButton } from '@fluentui/react';
import { useAppSelector, useAppDispatch } from '@src/models/hooks';
import { selectCallout, selectCurrent, selectInfo, updateCallout } from '@src/models/template';
import style from './style';
const { Item } = Stack;

const ModuleDetail: React.FC = () => {
  const dispatch = useAppDispatch();
  const callout = useAppSelector(selectCallout);
  const current = useAppSelector(selectCurrent);
  const info = useAppSelector(selectInfo);

  // 鼠标事件
  const handleMouseEvent = (e: any) => {
    dispatch(updateCallout(e.type !== 'mouseenter'));
  };

  return (
    <Callout
      gapSpace={-10}
      hidden={callout}
      hideOverflow
      calloutMaxHeight={480}
      isBeakVisible={false}
      preventDismissOnLostFocus={true}
      target={{
        current,
      }}
      directionalHint={12}
    >
      <Stack className={style.calloutContent} onMouseEnter={handleMouseEvent} onMouseLeave={handleMouseEvent}>
        <Item>
          <h3 className={style.moduleDetailTitle}>{info.name}</h3>
        </Item>
        <Item grow={1} shrink={1} className={style.moduleDetailItemStyle}>
          <div>
            <Label title="Description" className={style.moduleDetailLabel}>
              描述
            </Label>
            <div className={style.moduleDetailBody}>
              <p>{info.describe}</p>
            </div>
          </div>
        </Item>
        <Item grow={1} shrink={1} className={style.moduleDetailItemStyle}>
          <div>
            <Label title="Description" className={style.moduleDetailLabel}>
              创建人
            </Label>
            <div className={style.moduleDetailBody}>
              <p>{info.createdBy}</p>
            </div>
          </div>
        </Item>
        <Item grow={1} shrink={1} className={style.moduleDetailItemStyle}>
          <div>
            <Label title="Description" className={style.moduleDetailLabel}>
              镜像
            </Label>
            <div className={style.moduleDetailBody}>
              <p>{info.imagesName}</p>
            </div>
          </div>
        </Item>
        <Item grow={1} shrink={1} className={style.moduleDetailItemStyle}>
          <div>
            <Label title="Description" className={style.moduleDetailLabel}>
              上次修改时间
            </Label>
            <div className={style.moduleDetailBody}>
              <p>{info.lastChanged}</p>
            </div>
          </div>
        </Item>
        <Item grow={1} shrink={1} className={style.moduleDetailItemStyle}>
          <div>
            <Label title="Description" className={style.moduleDetailLabel}>
              版本
            </Label>
            <div className={style.moduleDetailBody}>
              <p>{info.version}</p>
            </div>
          </div>
        </Item>
        {info?.expand?.help_url ? (
          <Item grow={1} shrink={1} className={style.moduleButton}>
            <PrimaryButton
              onClick={() => {
                window.open(info.expand.help_url);
              }}
            >
              配置文档
            </PrimaryButton>
          </Item>
        ) : null}
      </Stack>
    </Callout>
  );
};

export default ModuleDetail;
