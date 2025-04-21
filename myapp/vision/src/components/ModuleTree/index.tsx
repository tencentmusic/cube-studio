import React, { useState, useEffect } from 'react';
import { Stack, SearchBox, Text, FocusZone, IconButton, Icon, Spinner, SpinnerSize, Callout } from '@fluentui/react';
import { useAppDispatch, useAppSelector } from '@src/models/hooks';
import { updateErrMsg } from '@src/models/app';
import { selectShow } from '@src/models/template';
import api from '@src/api';
import storage from '@src/utils/storage';
import { debounce } from '@src/utils';
import ModuleDetail from './components/ModuleDetail';
import ModuleItem from './components/ModuleItem';
import SearchItem from './components/SearchItem';
import style from './style';
import { useTranslation } from 'react-i18next';

const { Item } = Stack;

const ColorList: Array<{
  color: string,
  bg: string
}> = [
    {
      color: 'rgba(0,120,212,1)',
      bg: 'rgba(0,120,212,0.02)',
    }, {
      color: 'rgba(0,170,200,1)',
      bg: 'rgba(0,170,200,0.02)',
    }, {
      color: 'rgba(0,200,153,1)',
      bg: 'rgba(0,200,153,0.02)',
    }, {
      color: 'rgba(0,6,200,1)',
      bg: 'rgba(0,6,200,0.02)',
    }, {
      color: 'rgba(212,65,0,1)',
      bg: 'rgba(212,65,0,0.02)',
    }, {
      color: 'rgba(212,176,0,1)',
      bg: 'rgba(212,176,0,0.02)',
    },
  ]

const ModuleTree: React.FC = () => {
  const dispatch = useAppDispatch();
  const show = useAppSelector(selectShow);
  const [expandNodes, setExpandNodes] = useState(new Set()); // 记录展开的模板节点
  const [nodeMap, setNodeMap] = useState(new Map()); // 模板节点分类
  const [nodeCount, setNodeCount] = useState(0);
  const [loading, setLoading] = useState(false);
  const [showSearch, setShowSearch] = useState(false);
  const [searchResult, setSearchResult] = useState(new Map());

  const { t, i18n } = useTranslation();

  const handleTemplateData = (res: any) => {
    const dataSet = new Map();
    let count = 0;

    res.forEach((ele: any) => {
      if (ele.version !== 'Release') return;
      count += 1;
      const baseInfo = {
        id: ele.id,
        args: ele.args ? JSON.parse(ele.args) : {},
        name: ele.name,
        version: ele.version,
        describe: ele.describe,
        imagesName: ele.images.name,
        createdBy: ele.created_by.username,
        lastChanged: ele.changed_on,
        expand: JSON.parse(ele.expand),
        color: ColorList[ele.project.id % ColorList.length]
      };
      if (!dataSet.has(ele.project.id)) {
        dataSet.set(ele.project.id, {
          id: ele.project.id,
          title: ele.project.name,
          children: [baseInfo],
        });
      } else {
        dataSet.get(ele.project.id).children.push(baseInfo);
      }
    });

    setNodeCount(count);
    setNodeMap(dataSet);
  };

  // 获取任务模板
  const updateTemplateList = () => {
    setLoading(true);
    setNodeCount(0);
    api
      .job_template_modelview()
      .then((res: any) => {
        if (res?.status === 0 && res?.message === 'success') {
          handleTemplateData(res?.result.data);

          const currentTime = Date.now();
          storage.set('job_template', {
            update: currentTime,
            value: res?.result.data,
            expire: 1000 * 60 * 60 * 24, // 24小时更新一次
          });
        }
      })
      .catch(err => {
        if (err.response) {
          dispatch(updateErrMsg({ msg: err.response?.data?.message }));
        }
      })
      .finally(() => {
        setLoading(false);
      });
  };

  useEffect(() => {
    const jobTemplate = storage.get('job_template');

    if (jobTemplate && Date.now() - jobTemplate?.update < jobTemplate?.expire) {
      handleTemplateData(jobTemplate.value);
    } else {
      updateTemplateList();
    }
  }, []);

  return (
    <Item shrink>
      <div className={show ? style.showModuleTree : style.hideModuleTree}>
        <div className={style.treeContainer}>
          {/* 模板搜索 */}
          <Stack horizontal horizontalAlign="space-between">
            <SearchBox
              placeholder={t('搜索模板名称或描述')}
              role="search"
              className={style.searchBoxStyle}
              onChange={debounce((event, newValue) => {
                if (!newValue) {
                  setShowSearch(false);
                  return;
                }
                console.log(newValue);
                const temp = new Map();

                nodeMap.forEach((value, key) => {
                  const { children } = value;

                  if (children.length > 0) {
                    children.forEach((element: any) => {
                      if (element?.name?.indexOf(newValue) > -1 || element?.describe?.indexOf(newValue) > -1) {
                        if (temp.has(key)) {
                          temp.set(key, temp.get(key).concat(element));
                        } else {
                          temp.set(key, [element]);
                        }
                      }
                    });
                  }
                });
                setSearchResult(temp);
                setShowSearch(true);
              }, 300)}
              onBlur={() => {
                setTimeout(() => {
                  setShowSearch(false);
                }, 300);
              }}
            ></SearchBox>
            <Callout
              className={style.searchCallout}
              isBeakVisible={false}
              preventDismissOnLostFocus={true}
              hidden={!showSearch}
              calloutMaxHeight={300}
              target={`.ms-SearchBox`}
            >
              <Stack className={style.searchListStyle}>
                {Array.from(searchResult.keys()).length === 0 ? (
                  <div
                    style={{
                      textAlign: 'center',
                    }}
                  >
                    {t('暂无匹配')}
                  </div>
                ) : (
                    ''
                  )}
                {Array.from(searchResult.keys()).map((key: any) => {
                  const currentRes = searchResult.get(key);

                  return (
                    <React.Fragment key={key}>
                      {currentRes?.map((cur: any) => {
                        return (
                          <SearchItem
                            key={cur.id}
                            model={cur}
                            onClick={() => {
                              console.log(expandNodes);
                              if (!expandNodes.has(key)) {
                                expandNodes.add(key);
                                setExpandNodes(new Set(expandNodes));
                              }
                            }}
                          />
                        );
                      })}
                    </React.Fragment>
                  );
                })}
              </Stack>
            </Callout>
          </Stack>
          {/* 模板统计 & 手动刷新 */}
          <div className={style.summaryStyle}>
            <Text>{nodeCount} assets in total</Text>
            <FocusZone>
              <IconButton
                iconProps={{
                  iconName: 'Refresh',
                }}
                onClick={() => {
                  if (!loading) {
                    updateTemplateList();
                  }
                }}
              ></IconButton>
              <IconButton
                iconProps={{
                  iconName: 'Add',
                }}
                onClick={() => {
                  window.open('/frontend/train/train_template/job_template?isVisableAdd=true')
                }}
              ></IconButton>
            </FocusZone>
          </div>
          {/* 模板列表 */}
          <div className={style.moduleTreeStyle}>
            <div className={style.moduleTreeBody}>
              {loading ? (
                <Stack className={style.spinnerContainer}>
                  <Spinner size={SpinnerSize.large} label="Loading" />
                </Stack>
              ) : (
                  <ul className={style.moduleListStyle}>
                    {Array.from(nodeMap.keys()).map((key: any) => {
                      const curNode = nodeMap.get(key);
                      return (
                        <li key={key} className={style.moduleListItem}>
                          <div
                            role="button"
                            onClick={() => {
                              if (expandNodes.has(key)) {
                                expandNodes.delete(key);
                              } else {
                                expandNodes.add(key);
                              }
                              setExpandNodes(new Set(expandNodes));
                            }}
                          >
                            <div className={style.itemFolderNode}>
                              <Icon
                                iconName={expandNodes.has(key) ? 'FlickUp' : 'FlickLeft'}
                                styles={{
                                  root: {
                                    alignItems: expandNodes.has(key) ? 'baseline' : 'center',
                                  },
                                }}
                                className={style.listIconStyle}
                              />
                              {curNode.title}
                            </div>
                          </div>
                          {expandNodes.has(key) ? (
                            <ul role="group" style={{ paddingLeft: '0px' }}>
                              {curNode.children?.map((cur: any) => {
                                return (
                                  <li className={style.moduleListItem} key={cur.id}>
                                    <div role="button">
                                      <ModuleItem model={cur}></ModuleItem>
                                    </div>
                                  </li>
                                );
                              })}
                            </ul>
                          ) : null}
                        </li>
                      );
                    })}
                  </ul>
                )}
            </div>
          </div>
        </div>
      </div>
      <ModuleDetail></ModuleDetail>
    </Item>
  );
};

export default ModuleTree;
