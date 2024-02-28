/* eslint-disable react/display-name */
import React, { useEffect, useState } from 'react';
import {
  DetailsList,
  DetailsListLayoutMode,
  ShimmeredDetailsList,
  IStackStyles,
  SelectionMode,
  Stack,
  IColumn,
  TooltipHost,
  IconButton,
  PrimaryButton,
  Pivot,
  PivotItem,
  Dropdown,
  IDropdownOption,
  IDropdownStyles,
  Text,
} from '@fluentui/react';
import api from '@src/api';
import Section from '@src/components/Home/Section';
import { videoDemo } from '@src/static/home';
import { useAppDispatch, useAppSelector } from '@src/models/hooks';
import { getPipelineList, selectPipelineList, selectAll, getAllList } from '@src/models/pipeline';
import { useTranslation } from 'react-i18next';

const homeContainerStyle: IStackStyles = {
  root: {
    width: '100%',
    height: '100%',
    overflow: 'hidden',
  },
};

const dropdownStyles: Partial<IDropdownStyles> = {
  dropdown: { width: 100, marginLeft: 10, marginRight: 20 },
};
const options: IDropdownOption[] = [
  {
    key: '10',
    text: '10',
  },
  {
    key: '25',
    text: '25',
  },
  {
    key: '50',
    text: '50',
  },
  {
    key: '100',
    text: '100',
  },
];

const Home: React.FC = () => {
  const dispatch = useAppDispatch();
  const myPipeLine = useAppSelector(selectPipelineList);
  const myAll = useAppSelector(selectAll);
  const [videoList, setVideoList] = useState<any>([]);
  const [pipelineList, setPipelineList] = useState<any>([]);
  const [projectList, setProjectList] = useState<any>([]);
  const [page, setPage] = useState<number>(0);
  const [pageSize, setPageSize] = useState<number>(10);
  const [hasProject, setHasProject] = useState<boolean>(false);
  const [isEnd, setIsEnd] = useState<boolean>(false);

  const { t, i18n } = useTranslation();

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
      name: t('任务流'),
      fieldName: 'name',
      minWidth: 200,
      maxWidth: 350,
      data: 'string',
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
          dangerouslySetInnerHTML={{ __html: item.name }}
        >
        </span>
      ),
    },
    {
      key: 'describe',
      name: t('描述'),
      fieldName: 'describe',
      minWidth: 200,
      maxWidth: 300,
      data: 'string',
    },
    {
      key: 'changed_on',
      name: t('修改时间'),
      fieldName: 'changed_on',
      minWidth: 200,
      maxWidth: 300,
      data: 'string',
    },
    {
      key: 'project_id',
      name: t('项目组'),
      minWidth: 150,
      maxWidth: 200,
      onRender: (item: any) => {
        return <div>{projectList[item.project_id].name}</div>;
      },
    },
  ];

  // 跳转指定pipeline
  const goPipeline = (item: any) => {
    const url = `${window.location.origin}${location.pathname}?pipeline_id=${item?.id}`;
    window.open(`${window.location.origin}/frontend/showOutLink?url=${encodeURIComponent(url)}`, 'bank');
    // if (window.self === window.top) {
    //   window.location.href = `${window.location.origin}${location.pathname}?pipeline_id=${item?.id}`;
    // } else {
    //   window.parent.postMessage(
    //     {
    //       type: 'link',
    //       message: {
    //         pipelineId: item?.id,
    //       },
    //     },
    //     `${window.location.origin}`,
    //   );
    // }
  };

  // 删除指定pipeline
  const deletePipeline = (item: any) => {
    if (item?.id) {
      api.pipeline_modelview_delete(item.id).then((res: any) => {
        if (res.status === 0) {
          dispatch(getPipelineList());
        }
      });
    }
  };

  // 初始化
  useEffect(() => {
    setVideoList([
      {
        name: t('新人制作一个pipeline'),
        img: '/static/assets/images/ad/video-cover1-thumb.png',
        url: 'https://cube-studio.oss-cn-hangzhou.aliyuncs.com/cube-studio.mp4',
        type: 'video',
      }
    ]);
    api.pipeline_modelview_demo().then((res: any) => {
      if (res.status === 0) {
        const { result } = res;
        const pipelineDemo = result.map((ele: any) => {
          const cur = {
            id: ele.id,
            name: ele.describe,
            img: JSON.parse(ele.parameter).img || '',
            type: 'link',
          };
          return cur;
        });
        setPipelineList(pipelineDemo);
      }
    });
    api.project_all().then((res: any) => {
      if (res.status === 0) {
        setHasProject(true);
        const list: any = [];
        res.result.data.forEach((ele: any) => {
          if (ele.id) {
            list[ele.id] = ele;
          }
        });
        console.log('list', list);
        setProjectList(list);
        dispatch(getPipelineList());
      }
    });
  }, []);

  useEffect(() => {
    if (hasProject) {
      dispatch(
        getAllList({
          page,
          page_size: pageSize,
        }),
      );
    }
  }, [pageSize, hasProject, page]);

  useEffect(() => {
    if (myAll) {
      setIsEnd(myAll.length < pageSize);
    }
  }, [myAll]);

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
        <Section name={t('平台主要功能')} data={pipelineList} first={true}></Section>
        <Section name={t('新手视频')} data={videoList}></Section>
        <Stack
          styles={{
            root: {
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
                lineHeight: '1.1',
                fontSize: 20,
                fontWeight: 'bold',
              }}
            >
              {t('流水线')}
            </div>
          </Stack>
          <Pivot aria-label="Basic Pivot Example" defaultSelectedKey="1">
            <PivotItem
              headerText={t('我的')}
              headerButtonProps={{
                'data-order': 1,
                'data-title': 'My Files Title',
              }}
              itemKey="1"
            >
              <div>
                <DetailsList
                  items={myPipeLine}
                  columns={column.concat({
                    key: 'action',
                    name: t('操作'),
                    minWidth: 200,
                    maxWidth: 300,
                    onRender: (item: any) => {
                      return (
                        <div>
                          <TooltipHost content={t('删除')}>
                            <IconButton
                              onClick={() => {
                                deletePipeline(item);
                              }}
                              iconProps={{
                                iconName: 'Delete',
                                styles: {
                                  root: {
                                    color: 'red',
                                  },
                                },
                              }}
                            ></IconButton>
                          </TooltipHost>
                        </div>
                      );
                    },
                  })}
                  selectionMode={SelectionMode.none}
                  setKey="none"
                  layoutMode={DetailsListLayoutMode.fixedColumns}
                  isHeaderVisible={true}
                  compact={true}
                  styles={{
                    headerWrapper: {
                      '.ms-DetailsHeader': {
                        paddingTop: 0,
                      },
                    },
                    contentWrapper: {
                      lineHeight: '32px',
                    },
                  }}
                />
              </div>
            </PivotItem>
            <PivotItem headerText={t('协作')} itemKey="2">
              <div>
                <ShimmeredDetailsList
                  setKey="none"
                  isHeaderVisible={true}
                  items={myAll || []}
                  columns={column}
                  compact={!!myAll}
                  selectionMode={SelectionMode.none}
                  layoutMode={DetailsListLayoutMode.fixedColumns}
                  enableShimmer={!myAll}
                  detailsListStyles={{
                    headerWrapper: {
                      '.ms-DetailsHeader': {
                        paddingTop: 0,
                      },
                    },
                    contentWrapper: {
                      lineHeight: '32px',
                    },
                  }}
                  listProps={{
                    renderedWindowsAhead: 0,
                    renderedWindowsBehind: 0,
                  }}
                ></ShimmeredDetailsList>
                <Stack
                  horizontal
                  reversed
                  verticalAlign="center"
                  styles={{
                    root: {
                      marginTop: 20,
                    },
                  }}
                >
                  <PrimaryButton
                    text={t('下一页')}
                    styles={{ root: { marginRight: 10 } }}
                    disabled={isEnd}
                    onClick={() => {
                      if (isEnd || !myAll) return;
                      setPage(page + 1);
                    }}
                  ></PrimaryButton>
                  <PrimaryButton
                    text={t('上一页')}
                    styles={{ root: { marginRight: 10 } }}
                    disabled={page === 0}
                    onClick={() => {
                      if (page === 0 || !myAll) return;
                      setPage(page - 1);
                    }}
                  ></PrimaryButton>
                  <Dropdown
                    defaultSelectedKey={'10'}
                    placeholder={t('选择页数')}
                    options={options}
                    styles={dropdownStyles}
                    onChange={(e, opt) => {
                      setPage(0);
                      opt?.key && setPageSize(+opt.key);
                    }}
                  />
                  <Text
                    styles={{
                      root: {
                        fontSize: 14,
                        fontWeight: 600,
                      },
                    }}
                  >
                    {t('选择页数')}
                  </Text>
                </Stack>
              </div>
            </PivotItem>
          </Pivot>
        </Stack>
      </Stack>
    </Stack>
  );
};

export default Home;
