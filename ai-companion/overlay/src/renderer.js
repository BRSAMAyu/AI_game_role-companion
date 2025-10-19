const textElement = document.getElementById('overlay-text');
const STYLE_CLASSES = ['subtitle-default', 'subtitle-battle'];
let hideTimeout = null;

function applyStyle(style) {
  STYLE_CLASSES.forEach((cls) => textElement.classList.remove(cls));
  const className = style === 'battle' ? 'subtitle-battle' : 'subtitle-default';
  textElement.classList.add(className);
}

window.overlayAPI.onShowText(({ text, ms, style }) => {
  applyStyle(style);
  textElement.textContent = text;
  textElement.classList.add('visible');
  if (hideTimeout) {
    clearTimeout(hideTimeout);
  }
  const defaultDuration = style === 'battle' ? 2400 : 4000;
  hideTimeout = setTimeout(() => {
    textElement.classList.remove('visible');
  }, ms || defaultDuration);
});
