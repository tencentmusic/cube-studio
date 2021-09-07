import React, { useEffect, useState } from 'react';
import { DetailsList, DetailsListLayoutMode, IStackStyles, SelectionMode, Stack, IColumn } from '@fluentui/react';
import Section from '@src/components/Home/Section';
import { pipelineDemo, videoDemo } from '@src/static/home';
import { useAppDispatch, useAppSelector } from '@src/models/hooks';
import { getPipelineList, selectPipelineList } from '@src/models/pipeline';

const homeContainerStyle: IStackStyles = {
  root: {
    width: '100%',
    height: '100%',
    overflow: 'hidden',
  },
};

const Home: React.FC = () => {
  const dispatch = useAppDispatch();
  const myPipeLine = useAppSelector(selectPipelineList);
  const [videoList, setVideoList] = useState<any>([]);
  const [pipelineList, setPipelineList] = useState<any>([]);

  const column: IColumn[] = [
    {
      key: 'id',
      name: 'ID',
      fieldName: 'id',
      minWidth: 50,
      maxWidth: 100,
      data: 'number',
    },
    {
      key: 'name',
      name: '任务流',
      fieldName: 'name',
      minWidth: 200,
      maxWidth: 350,
      data: 'string',
      // eslint-disable-next-line react/display-name
      onRender: (item: any) => (
        <span
          style={{
            textDecoration: 'underline',
            color: '#005ccb',
            cursor: 'pointer',
          }}
          onClick={() => {
            goPipeline(item);
          }}
        >
          {item.name}
        </span>
      ),
    },
    {
      key: 'describe',
      name: '描述',
      fieldName: 'describe',
      minWidth: 200,
      maxWidth: 300,
      data: 'string',
    },
    {
      key: 'changed_on',
      name: '修改时间',
      fieldName: 'changed_on',
      minWidth: 200,
      maxWidth: 300,
      data: 'string',
    },
  ];

  const goPipeline = (item: any) => {
    if (window.self === window.top) {
      window.location.href = `${window.location.origin}${location.pathname}?pipeline_id=${item?.id}`;
    } else {
      window.parent.postMessage(
        {
          type: 'link',
          message: {
            pipelineId: item?.id,
          },
        },
        `${window.location.origin}`,
      );
    }
  };

  useEffect(() => {
    setVideoList(videoDemo);
    setPipelineList(pipelineDemo);
    dispatch(getPipelineList());
  }, []);

  return (
    <Stack className="home-container" styles={homeContainerStyle}>
      <Stack
        as="main"
        grow
        verticalFill
        styles={{
          root: {
            padding: '8px 14px',
            overflow: 'scroll',
          },
        }}
      >
        <Section name="平台主要功能" data={pipelineList} first={true}></Section>
        <Section name="新手视频" data={videoList}></Section>
        <Stack
          styles={{
            root: {
              width: '50%',
              marginTop: '16px !important',
              padding: '0 10px 24px',
            },
          }}
        >
          <Stack className="flex-section" horizontal horizontalAlign={'space-between'}>
            <div
              className="subtitle"
              style={{
                marginBottom: 8,
                height: 24,
                lineHeight: '22px',
                fontSize: 16,
                fontWeight: 600,
              }}
            >
              我的流水线
            </div>
          </Stack>
          <DetailsList
            items={myPipeLine}
            columns={column}
            selectionMode={SelectionMode.none}
            setKey="none"
            layoutMode={DetailsListLayoutMode.justified}
            isHeaderVisible={true}
          />
        </Stack>
      </Stack>
    </Stack>
  );
};

export default Home;
