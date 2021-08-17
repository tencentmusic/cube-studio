import { mergeStyles } from '@fluentui/react';

const moduleScroll = {
  '::-webkit-scrollbar': {
    width: 4,
  },
  '::-webkit-scrollbar-thumb': {
    minHeight: '15px',
    border: '6px solid transparent',
    backgroundClip: 'padding-box',
    backgroundColor: 'rgb(200, 200, 200)',
  },
};
const showModuleTree = mergeStyles({
  width: '320px',
  height: '100%',
  transition: 'all 0.35s ease 0s',
  overflowX: 'hidden',
  visibility: 'visible',
});
const hideModuleTree = mergeStyles({
  width: '0px',
  height: '100%',
  transition: 'width 0.35s ease 0s',
  overflowX: 'hidden',
  visibility: 'hidden',
});
const treeContainer = mergeStyles({
  width: '320px',
  boxSizing: 'border-box',
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  borderRight: '1px solid rgb(234, 234, 234)',
});
const searchBoxStyle = mergeStyles({
  margin: '12px 16px',
  padding: '1px 0px',
  color: 'rgb(1, 92, 218)',
  lineHeight: '32px',
  minHeight: '32px',
  flexGrow: '1',
});
const searchCallout = mergeStyles({
  '.ms-Callout-main': {
    overflowY: 'overlay',
    willChange: 'transform',
    ...moduleScroll,
  },
});
const searchListStyle = mergeStyles({
  width: '287px',
  padding: '8px 0',
});
const summaryStyle = mergeStyles({
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  padding: '8px 16px',
  borderTop: '1px solid rgb(218, 218, 218)',
});
const moduleTreeStyle = mergeStyles({
  position: 'relative',
  display: 'flex',
  flexDirection: 'column',
  overflowX: 'hidden',
  overflowY: 'auto',
  flexGrow: '1',
});
const moduleTreeBody = mergeStyles({
  position: 'relative',
  height: '100%',
});
const listIconStyle = mergeStyles({
  display: 'inline-flex',
  fontSize: '12px',
  lineHeight: '12px',
  marginRight: '4px',
  height: '6px',
  color: 'rgb(55, 55, 55)',
  userSelect: 'none',
});
const spinnerContainer = mergeStyles({
  width: '100%',
  height: '100%',
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
});
const moduleListStyle = mergeStyles({
  padding: 0,
  margin: '0 auto',
  overflowY: 'overlay',
  height: '100%',
  willChange: 'transform',
  ...moduleScroll,
});
const moduleListItem = mergeStyles({
  listStyle: 'none',
  outline: 'none',
});
const itemFolderNode = mergeStyles({
  display: 'flex',
  alignItems: 'center',
  cursor: 'pointer',
  padding: '7px 15px',
  border: '1px solid transparent',
  height: 16,
  fontFamily: 'Segoe UI,sans-serif',
  fontSize: 12,
  lineHeight: 16,
  fontWeight: 600,
  color: 'black',
  '&:hover': {
    backgroundColor: '#eaeaea',
  },
});

export default {
  showModuleTree,
  hideModuleTree,
  treeContainer,
  searchBoxStyle,
  searchCallout,
  searchListStyle,
  summaryStyle,
  moduleTreeStyle,
  moduleTreeBody,
  listIconStyle,
  spinnerContainer,
  moduleListStyle,
  moduleListItem,
  itemFolderNode,
};
