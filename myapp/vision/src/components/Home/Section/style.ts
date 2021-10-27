import { mergeStyles } from '@fluentui/merge-styles';

const sectionStyles = mergeStyles({
  marginBottom: 16,
  padding: '0 10px 24px 10px',
  borderBottom: '1px solid #dadada',
  selectors: {
    '.subtitle': {
      marginBottom: 20,
      height: 24,
      lineHeight: '1.1',
      fontSize: 20,
      fontWeight: 'bold',
      fontFamily: '"Raleway","Helvetica Neue",Helvetica,Arial,sans-serif;',
    },
    '.expand-button': {
      color: '#005cd2',
      marginRight: 24,
      fontSize: 14,
      marginBottom: 12,
    },
  },
});

const sampleStyles = mergeStyles({
  minHeight: 186,
  position: 'relative',
  display: 'flex',
  flexWrap: 'wrap',
});
const sampleHide = mergeStyles({
  height: 186,
  overflow: 'hidden',
});

const sampleCardStyle = mergeStyles({
  width: 200,
  marginRight: 24,
  marginBottom: 24,
});

const cardContainer = mergeStyles({
  width: 200,
  height: 124,
  borderRadius: 2,
  display: 'flex',
  flexDirection: 'column',
  cursor: 'pointer',
  backgroundColor: '#fff',
  border: '1px solid #dadada',
  selectors: {
    ':hover': {
      boxShadow: 'rgb(0 0 0 / 18%) 0px 6.4px 14.4px 0px, rgb(0 0 0 / 22%) 0px 1.2px 3.6px 0px',
    },
  },
});

const addIconStyles = mergeStyles({
  fontSize: 38,
  padding: '42px 0 29px',
  cursor: 'pointer',
  textAlign: 'center',
  color: '#005cd2',
});

const sampleImgStyles = mergeStyles({
  height: '100%',
  backgroundRepeat: 'no-repeat',
  backgroundSize: 'cover',
  selectors: {
    img: {
      width: '100%',
      height: '100%',
    },
  },
});

const cardTitleStyles = mergeStyles({
  marginTop: 16,
  textAlign: 'center',
  cursor: 'pointer',
  whiteSpace: 'nowrap',
  display: 'flex',
  flexFlow: 'column nowrap',
  width: 'auto',
  height: 'auto',
  textOverflow: 'ellipsis',
});

const videoPlayerStyles = mergeStyles({
  backgroundColor: '#000',
  width: 800,
  height: 500,
});

export default {
  sectionStyles,
  sampleStyles,
  sampleHide,
  sampleCardStyle,
  cardContainer,
  addIconStyles,
  sampleImgStyles,
  cardTitleStyles,
  videoPlayerStyles,
};
