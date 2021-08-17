import { mergeStyles } from '@fluentui/react';
const settingContainer = mergeStyles({
  display: 'flex',
  flexDirection: 'column',
  width: 340,
  position: 'absolute',
  top: 0,
  right: 0,
  bottom: 0,
  height: 'auto',
  backgroundColor: '#fff',
  borderLeft: '1px solid rgb(234, 234, 234)',
  boxShadow:
    'rgb(0 0 0 / 18%) 0px 1.6px 3.6px 0px, rgb(0 0 0 / 22%) 0px 0.3px 0.9px 0px',
  zIndex: 999,
});
const settingHeader = mergeStyles({
  display: 'flex',
  width: '100%',
  alignItems: 'center',
  justifyContent: 'space-between',
  color: 'balck',
  boxSizing: 'border-box',
  padding: '10px 20px',
});
const headerTitle = mergeStyles({
  fontSize: 20,
  fontWeight: 600,
  textOverflow: 'ellipsis',
  overflow: 'hidden',
  whiteSpace: 'nowrap',
  color: 'black',
});
const settingContent = mergeStyles({
  flex: '1 1 0%',
  overflowY: 'auto',
  '&::-webkit-scrollbar': {
    width: 4,
  },
  '&::-webkit-scrollbar-thumb': {
    minHeight: '15px',
    border: '6px solid transparent',
    backgroundClip: 'padding-box',
    backgroundColor: 'rgb(200, 200, 200)',
  },
});
const contentWrapper = mergeStyles({
  padding: '0 20px 20px',
  overflowX: 'hidden',
  wordBreak: 'break-word',
  userSelect: 'text',
  borderTop: '1px solid #eaeaea',
});
const splitLine = mergeStyles({
  margin: '10px -20px 0 -20px',
  borderTop: '1px solid #eaeaea',
});

export default {
  settingContainer,
  settingHeader,
  headerTitle,
  settingContent,
  contentWrapper,
  splitLine,
};
