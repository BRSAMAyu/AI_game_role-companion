const textElement = document.getElementById('overlay-text');
let hideTimeout = null;

window.overlayAPI.onShowText(({ text, ms }) => {
  textElement.textContent = text;
  textElement.classList.add('visible');
  if (hideTimeout) {
    clearTimeout(hideTimeout);
  }
  hideTimeout = setTimeout(() => {
    textElement.classList.remove('visible');
  }, ms || 4000);
});
