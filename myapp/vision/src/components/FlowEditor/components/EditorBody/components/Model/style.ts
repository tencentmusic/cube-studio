import { mergeStyles } from '@fluentui/react';
const modelContainer = mergeStyles({
  display: 'flex',
  flexDirection: 'column',
  width: 350,
  position: 'absolute',
  top: 0,
  right: 0,
  bottom: 0,
  height: 'auto',
  backgroundColor: '#fff',
  borderLeft: '1px solid rgb(234, 234, 234)',
  boxShadow: 'rgb(0 0 0 / 18%) 0px 1.6px 3.6px 0px, rgb(0 0 0 / 22%) 0px 0.3px 0.9px 0px',
  zIndex: 998,
});
const modelHeader = mergeStyles({
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
  textOverflow: 'ellipsis',
  overflow: 'hidden',
  whiteSpace: 'nowrap',
  color: 'black',
  fontWeight: 600,
});
const modelContent = mergeStyles({
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
const settingControl = mergeStyles({
  height: 'auto',
  display: 'flex',
  flexDirection: 'column',
  paddingTop: 20,
});
const saveButton = mergeStyles({
  alignSelf: 'center',
});
const debugButton = mergeStyles({
  display: 'flex',
  flexDirection: 'row',
  justifyContent: 'space-evenly',
  paddingTop: 20,
});
const templateConfig = mergeStyles({
  color: 'rgb(96, 94, 92)',
  fontSize: 10,
  paddingTop: 5,
  paddingLeft: 2,
  '> a': {
    textDecoration: 'none',
  },
});
const argsDescription = mergeStyles({
  color: 'rgb(96, 94, 92)',
  fontSize: 10,
  '> a': {
    textDecoration: 'none',
  },
});
const textLabelStyle = mergeStyles({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  fontWeight: 600,
  color: 'rgb(50, 49, 48)',
  boxSizing: 'border-box',
  boxShadow: 'none',
  margin: '0px',
  padding: '5px 0px',
  overflowWrap: 'break-word',
  lineHeight: '30px',
});

export default {
  modelContainer,
  modelHeader,
  headerTitle,
  modelContent,
  contentWrapper,
  splitLine,
  settingControl,
  saveButton,
  debugButton,
  templateConfig,
  argsDescription,
  textLabelStyle,
};
